"""Run an upstream trainer with a recorded seed and resolved Gin config."""
import argparse
import importlib
import os
from pathlib import Path
import sys

base=Path('/root/genrec-reproduction')
repo=base/'genrec'
sys.path.insert(0,str(repo))
parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['rqvae','tiger'])
parser.add_argument('split',choices=['beauty','sports','toys','home'])
args=parser.parse_args()
os.chdir(repo)
os.environ['PYTHONHASHSEED']='42'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['WANDB_MODE']='disabled'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='6'
import gin
import torch
from genrec.modules.utils import parse_config
from genrec.trainers.trainer_utils import set_seed
trainer=importlib.import_module(f'genrec.trainers.{args.stage}_trainer')
sys.argv=[sys.argv[0],f'config/tiger/amazon/{args.stage}.gin','--split',args.split,
          '--gin',f"MODEL_HUB_SENTENCE_T5_XL='{base}/models_hub/sentence-t5-xl'",
          '--gin','train.wandb_logging=False']
parse_config()
set_seed(42,deterministic=True)
torch.set_num_threads(6)
resolved=base/'manifests'/f'{args.split}-{args.stage}-resolved.gin'
resolved.write_text(gin.config_str())
print(f'Stage={args.stage}, split={args.split}, seed=42, config={resolved}',flush=True)
trainer.train()
