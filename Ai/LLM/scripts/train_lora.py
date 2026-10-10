"""Windows-native LoRA/QLoRA with explicit assistant-only labels."""
import argparse, json, math, os, random, re, sys, time, traceback, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if (ROOT/'runtime/qlora-deps').is_dir():sys.path.insert(0,str(ROOT/'runtime/qlora-deps'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTHONUTF8']='1'
import torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,BitsAndBytesConfig
from peft import LoraConfig,get_peft_model,PeftModel,prepare_model_for_kbit_training
from src.chatml import encode_chatml
from src.sft_tasks import training_prompt
from src.training_loss import assistant_only_loss

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--run',required=True)
    parser.add_argument('--resume');parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--max-length',type=int,default=1024)
    parser.add_argument('--data',default='data/pilot_v2');parser.add_argument('--init-adapter');parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--profile',choices=['baseline','natural-v1','natural-v2'],default='baseline')
    parser.add_argument('--prompt-kind',choices=['legacy','character'],default='legacy')
    parser.add_argument('--assistant-logits-only',action='store_true')
    parser.add_argument('--quantization',choices=['none','nf4'],default='none')
    parser.add_argument('--rank',type=int,default=8);parser.add_argument('--alpha',type=int,default=32)
    parser.add_argument('--grad-accum',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--warmup-ratio',type=float,default=0)
    parser.add_argument('--speech-loss-weight',type=float,default=1)
    args=parser.parse_args(); dest=ROOT/'runs'/args.run
    if not 1<=args.epochs<=5: raise ValueError('Epochs must be 1..5')
    if args.grad_accum < 1 or args.batch_size < 1 or args.rank < 1 or not 0 <= args.warmup_ratio < 1:raise ValueError('Invalid training configuration')
    if args.speech_loss_weight<1 or args.speech_loss_weight!=1 and not args.assistant_logits_only:
        raise ValueError('Speech weighting requires assistant-selected logits and weight >= 1')
    if dest.exists(): raise RuntimeError('Run exists; choose a new run name, including when resuming')
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'pid').write_text(str(os.getpid()))
    log=(dest/'metrics.jsonl').open('a',encoding='utf-8')
    def emit(kind,**kw):
        row={'event':kind,'time':time.time(),**kw};line=json.dumps(row,ensure_ascii=False)
        print(line,flush=True);log.write(line+'\n');log.flush()
    def memory():
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
        return result.stdout.strip()
    torch.manual_seed(42);random.seed(42);torch.set_num_threads(8)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen3.5-9b',local_files_only=True)
    def read(split):
        rows=[json.loads(x) for x in (ROOT/args.data/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        batches=[]
        for row in rows:
            target=json.dumps(row['answer'],ensure_ascii=False,separators=(',',':'))
            encoded=encode_chatml(tok,training_prompt(row,args.profile,args.prompt_kind),
                target,args.max_length)
            ids,labels=encoded['input_ids'],encoded['labels']
            batch={'input_ids':torch.tensor([ids]),'labels':torch.tensor([labels]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long)}
            if args.speech_loss_weight!=1:
                offsets=tok(target+'<|im_end|>',add_special_tokens=False,return_offsets_mapping=True)
                assert offsets['input_ids']==ids[encoded['prompt_tokens']:]
                weights=[0.]*encoded['prompt_tokens']+[1.]*encoded['target_tokens']
                match=re.search(r'"speech":("(?:[^"\\]|\\.)*")',target)
                if match:
                    left,right=match.span(1)
                    for position,(a,b) in enumerate(offsets['offset_mapping']):
                        if a<right-1 and b>left+1:weights[encoded['prompt_tokens']+position]=args.speech_loss_weight
                batch['loss_weights']=torch.tensor([weights],dtype=torch.float32)
            batches.append(batch)
        return batches
    train=read('train');val=read('validation')
    if args.smoke:train=sorted(train,key=lambda b:b['input_ids'].shape[1],reverse=True)[:8];val=val[:2]
    emit('load_start',train_count=len(train),validation_count=len(val))
    quantization = None
    if args.quantization == 'nf4':
        quantization = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=['model.visual','visual','lm_head'])
    model=Qwen3_5ForConditionalGeneration.from_pretrained(ROOT/'models/qwen3.5-9b',dtype=torch.bfloat16,
             device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,quantization_config=quantization)
    if quantization is not None:
        model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
        import bitsandbytes
        quantized_layers=sum(isinstance(m,bitsandbytes.nn.Linear4bit) for m in model.modules())
        if not quantized_layers:raise RuntimeError('NF4 requested but no quantized layers loaded')
        emit('quantized',format='nf4',double_quant=True,compute_dtype='bfloat16',linear4bit_layers=quantized_layers,
             allocated_gib=torch.cuda.memory_allocated()/2**30)
    targets=[n for n,m in model.named_modules() if n.startswith('model.language_model.layers.') and isinstance(m,torch.nn.Linear)]
    if args.resume and args.init_adapter:raise ValueError('Resume and initialization are mutually exclusive')
    if args.resume or args.init_adapter:model=PeftModel.from_pretrained(model,ROOT/(args.resume or args.init_adapter),is_trainable=True)
    else:model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=args.alpha,lora_dropout=0.05,target_modules=targets,task_type='CAUSAL_LM'))
    model.config.use_cache=False
    if hasattr(model.config,'text_config'):model.config.text_config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert parameters and all('lora_' in n and 'visual' not in n for n,p in parameters)
    (dest/'trainable_parameters.json').write_text(json.dumps({n:p.numel() for n,p in parameters},indent=2))
    conf={'framework':'transformers+peft','model':json.loads((ROOT/'config/qwen3.5-9b.json').read_text()),
          'bf16':True,'quantization':None if quantization is None else quantization.to_dict(),
          'rank':next(iter(model.peft_config.values())).r,'alpha':next(iter(model.peft_config.values())).lora_alpha,
          'batch':args.batch_size,'grad_accum':args.grad_accum,'lr':args.lr,'warmup_ratio':args.warmup_ratio,
          'speech_loss_weight':args.speech_loss_weight,
          'epochs':1 if args.smoke else args.epochs,'max_length':args.max_length,'trainable':sum(p.numel() for _,p in parameters),
          'data_manifest':json.loads((ROOT/args.data/'manifest.json').read_text(encoding='utf-8')),'smoke':args.smoke,
          'data':args.data,'init_adapter':args.init_adapter,'runtime_guidance':True,'shuffle_seed':42,
          'prompt_profile':args.profile,'prompt_kind':args.prompt_kind,'assistant_logits_only':args.assistant_logits_only,
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'base_revision':json.loads((ROOT/'config/qwen3.5-9b.json').read_text())['revision']}
    (dest/'config.json').write_text(json.dumps(conf,ensure_ascii=False,indent=2),encoding='utf-8')
    optimizer=torch.optim.AdamW([p for _,p in parameters],lr=args.lr)
    total_updates=math.ceil(math.ceil(len(train)/args.batch_size)/args.grad_accum)*conf['epochs']
    warmup_updates=max(1,int(total_updates*args.warmup_ratio)) if args.warmup_ratio else 0
    def learning_rate(step):
        if warmup_updates and step <= warmup_updates:return args.lr*step/warmup_updates
        if args.warmup_ratio:return args.lr*max(0.1,(total_updates-step+1)/max(1,total_updates-warmup_updates))
        return args.lr
    start_epoch=0;steps=0
    if args.resume:
        previous=json.loads((ROOT/args.resume).parent.joinpath('config.json').read_text(encoding='utf-8'))
        if previous['data_manifest']['sha256']!=conf['data_manifest']['sha256']:raise ValueError('Resume data changed')
        saved=torch.load(ROOT/args.resume/'optimizer.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(saved['optimizer']);start_epoch=saved['epoch'];steps=saved['steps']
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    def forward_loss(batch):
        if isinstance(batch,list):
            width=max(b['input_ids'].shape[1] for b in batch)
            ids=torch.full((len(batch),width),tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,dtype=torch.long)
            labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids);weights=torch.zeros_like(ids,dtype=torch.float32)
            for index,item in enumerate(batch):
                length=item['input_ids'].shape[1]
                ids[index,:length]=item['input_ids'][0];labels[index,:length]=item['labels'][0];mask[index,:length]=1
                if 'loss_weights' in item:weights[index,:length]=item['loss_weights'][0]
            batch={'input_ids':ids,'labels':labels,'attention_mask':mask}
            if args.speech_loss_weight!=1:batch['loss_weights']=weights
        tensors={k:v.to('cuda') for k,v in batch.items()}
        if args.assistant_logits_only:return assistant_only_loss(model,tensors)
        return model(**tensors,use_cache=False).loss
    def evaluate():
        model.eval();losses=[]
        with torch.no_grad():
            for batch in val:
                loss=forward_loss(batch)
                losses.append(float(loss));del loss
                torch.cuda.empty_cache()
        model.train();return sum(losses)/len(losses)
    try:
        emit('loaded',trainable=conf['trainable'],allocated_gib=torch.cuda.memory_allocated()/2**30)
        choices=[]
        if args.resume:
            previous_metrics=(ROOT/args.resume).parent/'metrics.jsonl'
            choices=[{'checkpoint':r['checkpoint'],'validation_loss':r['loss']} for r in
                     (json.loads(line) for line in previous_metrics.read_text().splitlines()) if r['event']=='validation']
        best=min((r['validation_loss'] for r in choices),default=math.inf)
        for epoch in range(start_epoch,conf['epochs']):
            model.train();optimizer.zero_grad(set_to_none=True)
            if args.batch_size==1:
                order=list(range(len(train)));random.Random(42+epoch).shuffle(order)
            else:
                indices=sorted(range(len(train)),key=lambda idx:train[idx]['input_ids'].shape[1])
                order=[indices[i:i+args.batch_size] for i in range(0,len(indices),args.batch_size)]
                random.Random(42+epoch).shuffle(order)
            for i,idx in enumerate(order):
                batch=[train[j] for j in idx] if isinstance(idx,list) else train[idx]
                window=min(args.grad_accum,len(order)-(i//args.grad_accum)*args.grad_accum)
                loss=forward_loss(batch)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                value=float(loss.detach());(loss/window).backward();del loss
                emit('backward',epoch=epoch+1,sample=i+1,examples=len(idx) if isinstance(idx,list) else 1,loss=value,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,device_used_total_mib=memory())
                if (i+1)%args.grad_accum==0 or i+1==len(order):
                    grads=[p.grad for _,p in parameters if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('Nonfinite/missing gradient')
                    norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in parameters],1.0))
                    rate=learning_rate(steps+1)
                    for group in optimizer.param_groups:group['lr']=rate
                    optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
                    emit('optimizer_update',step=steps,gradient_norm=norm,learning_rate=rate)
                    del grads
                # Varying lengths fragmented WDDM allocations in the first smoke run.
                # Release unused blocks to prevent shared-memory spill, preserving live tensors.
                torch.cuda.empty_cache()
            val_loss=evaluate();checkpoint=dest/f'checkpoint-epoch-{epoch+1}'
            model.save_pretrained(checkpoint);tok.save_pretrained(checkpoint)
            torch.save({'optimizer':optimizer.state_dict(),'epoch':epoch+1,'steps':steps,
                        'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},checkpoint/'optimizer.pt')
            choices.append({'checkpoint':str(checkpoint),'validation_loss':val_loss})
            emit('validation',epoch=epoch+1,loss=val_loss,checkpoint=str(checkpoint))
            if (dest/'stop.request').exists():emit('stop_requested',after_epoch=epoch+1);break
            if val_loss<best:best=val_loss
            elif epoch>=1:emit('early_stop',reason='validation worsened');break
        selected=min(choices,key=lambda x:x['validation_loss'])
        (dest/'selection.json').write_text(json.dumps({'selected':selected,'candidates':choices,
               'basis':'validation loss; dialogue comparison required before promotion'},indent=2))
        # Actually reload adapter into the base and verify identical deterministic validation.
        del optimizer,parameters
        base_model=model.unload();del model
        torch.cuda.empty_cache()
        model=PeftModel.from_pretrained(base_model,selected['checkpoint']);model.eval()
        reloaded=evaluate()
        if abs(reloaded-selected['validation_loss'])>0.01:raise RuntimeError('Reload validation differs')
        emit('complete',optimizer_steps=steps,reload_validation_loss=reloaded,
             peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    except Exception as e:
        emit('failed',error=repr(e),allocated_gib=torch.cuda.memory_allocated()/2**30,reserved_gib=torch.cuda.memory_reserved()/2**30)
        traceback.print_exc();raise
    finally:log.close()

if __name__=='__main__':
    from filelock import FileLock
    with FileLock(str(ROOT/'runtime/gpu.lock'),timeout=0):main()
