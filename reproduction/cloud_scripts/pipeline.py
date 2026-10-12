"""Persistent four-dataset reproduction, final evaluation, and result comparison."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE=Path('/root/genrec-reproduction')
REPO=BASE/'genrec'
PYTHON=BASE/'.venv/bin/python'
SPLITS=['beauty','sports','toys','home']
EXPECTED={'beauty':[.0419,.0644,.0282,.0354], 'sports':[.0236,.0377,.0150,.0195],
          'toys':[.0340,.0521,.0214,.0272], 'home':[.0145,.0231,.0096,.0123]}
KEYS=['Recall@5','Recall@10','NDCG@5','NDCG@10']
status={'state':'waiting_for_setup','completed':[],'split':None,'stage':None,'pid':os.getpid(),
        'source_commit':'fb4bd1a1832df0312580a705d3440d1988856e31','seed':42,
        'started_at':datetime.now(timezone.utc).isoformat()}

def read_status(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return {'state':'pending'}

def save_status(**changes):
    status.update(changes)
    status['updated_at']=datetime.now(timezone.utc).isoformat()
    temporary=BASE/'status.json.tmp'
    temporary.write_text(json.dumps(status,indent=2))
    temporary.replace(BASE/'status.json')
    print(json.dumps(status),flush=True)

def wait_for_setup():
    manifests=['environment-status.json','model-status.json']
    while True:
        states={}
        for filename in manifests:
            path=BASE/'manifests'/filename
            states[filename]=read_status(path)
        failed={name:record for name,record in states.items() if record['state']=='failed'}
        if failed:
            raise RuntimeError('Setup failed: '+json.dumps(failed))
        if all(record['state']=='ready' for record in states.values()):
            return
        time.sleep(30)

def wait_for_dataset(split):
    directory=REPO/'dataset/amazon/raw'/split
    while True:
        if directory.exists() and len(list(directory.glob('*.verified.json')))==2:
            return
        overall=BASE/'manifests/data-status.json'
        if read_status(overall).get('state')=='failed':
            raise RuntimeError('Dataset acquisition failed; see logs/download-data.log')
        time.sleep(30)

def run_stage(split,stage,command):
    save_status(state='running',split=split,stage=stage)
    with (BASE/'logs'/f'{split}-{stage}.log').open('a') as log:
        log.write('\nRun started: '+datetime.now(timezone.utc).isoformat()+'\n')
        log.flush()
        subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)

def comparison():
    rows=[]
    for split in SPLITS:
        save_status(state='waiting_for_data',split=split,stage='download')
        wait_for_dataset(split)
        result=json.loads((BASE/'results'/f'{split}.json').read_text())
        row={'dataset':split,'best_epoch':result['best_epoch']}
        for key,expected in zip(KEYS,EXPECTED[split]):
            row[key]=result['metrics'][key]
            row[f'README_{key}']=expected
            row[f'delta_{key}']=result['metrics'][key]-expected
        rows.append(row)
    with (BASE/'results/comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    content=['# TIGER reproduction','',f"Source commit: {status['source_commit']}; seed: 42.",
             '','Validation-selected checkpoint; upstream batch-averaged test metrics.',
             'V100 evaluation uses FP32 because native BF16 is unavailable.',
             'Semantic-ID collisions and user-weighted metrics are recorded in each JSON.',
             '','| Dataset | R@5 | R@10 | N@5 | N@10 | README R@5 | README R@10 | README N@5 | README N@10 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[row[key] for key in KEYS]+[row['README_'+key] for key in KEYS]
        content.append('| '+row['dataset']+' | '+' | '.join(f'{value:.6f}' for value in values)+' |')
    (BASE/'results/comparison.md').write_text('\n'.join(content)+'\n')

def main():
    (BASE/'results').mkdir(exist_ok=True)
    save_status()
    wait_for_setup()
    env=os.environ
    env.update(PYTHONHASHSEED='42',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='6',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for split in SPLITS:
        result_path=BASE/'results'/f'{split}.json'
        if result_path.exists() and json.loads(result_path.read_text()).get('state')=='complete':
            status['completed'].append(split)
            continue
        save_status(state='waiting_for_data',split=split,stage='download')
        wait_for_dataset(split)
        checkpoint=REPO/'out/tiger/amazon'/split/'rqvae/checkpoint_epoch_4999.pt'
        if not checkpoint.exists():
            run_stage(split,'rqvae',[str(PYTHON),'-u',str(BASE/'run_stage.py'),'rqvae',split])
        assert checkpoint.exists(), 'Final RQVAE checkpoint missing'
        run_stage(split,'tiger',[str(PYTHON),'-u',str(BASE/'run_stage.py'),'tiger',split])
        run_stage(split,'evaluation',[str(PYTHON),'-u',str(BASE/'evaluate_tiger.py'),
            'config/tiger/amazon/tiger.gin','--split',split,
            '--gin',f"MODEL_HUB_SENTENCE_T5_XL='{BASE}/models_hub/sentence-t5-xl'",
            '--gin','train.wandb_logging=False'])
        status['completed'].append(split)
        save_status()
    comparison()
    save_status(state='complete',split=None,stage=None)

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        save_status(state='failed',error=str(error))
        traceback.print_exc()
        sys.exit(1)
