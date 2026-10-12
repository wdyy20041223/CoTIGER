"""Evaluate the validation-selected checkpoint and retain both metric averages."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

BASE=Path('/root/genrec-reproduction')
REPO=BASE/'genrec'
sys.path.insert(0,str(REPO))
os.chdir(REPO)
import gin
import torch
from torch.utils.data import DataLoader
from genrec.models.tiger import Tiger
from genrec.data.amazon import AmazonSeqDataset
from genrec.modules.utils import parse_config
from genrec.trainers.tiger_trainer import train, t5_collate, calculate_pos_index, recall_at_k, ndcg_at_k
from genrec.trainers.trainer_utils import set_seed

args=parse_config()
set_seed(42,deterministic=True)
torch.set_num_threads(6)
split=args.split
device=torch.device('cuda')
def binding(key):
    return gin.query_parameter('train.'+key)

config={k:binding(k) for k in ['num_layers','num_decoder_layers','d_model','d_ff',
                              'num_heads','d_kv','dropout_rate','feed_forward_proj','sem_id_dim']}
config.update(vocab_size=binding('codebook_size')*binding('sem_id_dim')+1,
              pad_token_id=0,eos_token_id=0)
out=REPO/'out/tiger/amazon'/split
checkpoint=out/'best_model.pt'
model=Tiger(config).to(device)
model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
model.eval()
dataset=AmazonSeqDataset(root='dataset/amazon',train_test_split='test',
                         max_seq_len=binding('max_seq_len'),
                         pretrained_rqvae_path=binding('pretrained_rqvae_path'))
codebook_size=binding('codebook_size')
sem_id_dim=binding('sem_id_dim')
max_history=binding('max_seq_len')*sem_id_dim
beams=binding('beam_size')
loader=DataLoader(dataset,batch_size=binding('infer_batch_size'),shuffle=False,num_workers=4,
                  pin_memory=True,collate_fn=lambda batch:t5_collate(batch,codebook_size,sem_id_dim,max_history))
totals={f'{metric}@{k}':0.0 for metric in ['Recall','NDCG'] for k in [5,10]}
batch_metrics={k:[] for k in totals}
processed=0
invalid=0
all_sids={tuple(c+idx*codebook_size+1 for idx,c in enumerate(sid)) for sid in dataset.sem_ids_list}
started=time.monotonic()
with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported(including_emulation=False)):
    for batch in loader:
        size=batch['input_ids'].shape[0]
        preds=model.generate(input_ids=batch['input_ids'].to(device),
                             attention_mask=batch['attention_mask'].to(device),num_beams=beams)
        preds=preds[:,1:].reshape(size,beams,-1).cpu()
        matches=calculate_pos_index(preds,batch['labels'],maxk=beams)
        for k in [5,10]:
            for name,fn in [('Recall',recall_at_k),('NDCG',ndcg_at_k)]:
                value=fn(matches,k)
                totals[f'{name}@{k}']+=value*size
                batch_metrics[f'{name}@{k}'].append(value)
        invalid+=sum(tuple(sid) not in all_sids for sid in preds[:,:10].reshape(-1,sem_id_dim).tolist())
        processed+=size
assert processed==len(dataset)
upstream_metrics={k:sum(v)/len(v) for k,v in batch_metrics.items()}
weighted_metrics={k:v/processed for k,v in totals.items()}
sid_counts=Counter(map(tuple,dataset.sem_ids_list))
log_text=(BASE/'logs'/f'{split}-tiger.log').read_text(errors='replace')
finished=re.findall(r'Training done\. Best Valid R@10=([0-9.]+) at epoch (\d+)',log_text)
assert finished,'Trainer did not finish normally'
best_recall,best_epoch=finished[-1]
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
result={'split':split,'seed':42,'state':'complete','best_epoch':int(best_epoch),
        'best_valid_recall10_logged':float(best_recall), 'users':processed,'items':len(dataset.sem_ids_list),
        'checkpoint':str(checkpoint),'checkpoint_sha256':digest,
        'metrics':upstream_metrics,'user_weighted_metrics':weighted_metrics,
        'metric_semantics':'Semantic-ID match, upstream three-code recipe with disambiguation disabled',
        'semantic_id_collisions':{'unique_sids':len(sid_counts),
            'colliding_groups':sum(count>1 for count in sid_counts.values()),
            'affected_items':sum(count for count in sid_counts.values() if count>1)},
        'invalid_generated_top10_fraction':invalid/(processed*10),
        'precision':'bf16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'fp32',
        'evaluation_seconds':time.monotonic()-started}
path=BASE/'results'/f'{split}.json'
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
