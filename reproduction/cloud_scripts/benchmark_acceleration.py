"""Bounded acceleration probe on real Beauty batches; never writes model weights."""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

BASE = Path('/root/genrec-reproduction')
REPO = BASE / 'genrec'
OUT = BASE / 'acceleration'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='6')
sys.path.insert(0, str(REPO))
os.chdir(REPO)
import gin
import torch
from genrec.models.tiger import Tiger
from genrec.data.amazon import AmazonSeqDataset
from genrec.modules.utils import parse_config
from genrec.trainers.tiger_trainer import train, t5_collate, calculate_pos_index, recall_at_k, ndcg_at_k
from genrec.trainers.trainer_utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--compile', action='store_true')
parser.add_argument('--buckets', action='store_true')
parser.add_argument('--steps', type=int, default=30)
args = parser.parse_args()
sys.argv = [sys.argv[0], 'config/tiger/amazon/tiger.gin', '--split', 'beauty',
            '--gin', f"MODEL_HUB_SENTENCE_T5_XL='{BASE}/models_hub/sentence-t5-xl'",
            '--gin', 'train.wandb_logging=False']
parse_config()
set_seed(42, deterministic=True)
torch.set_num_threads(6)
OUT.mkdir(exist_ok=True)
report = {'state': 'running', 'phase': 'data', 'seed': 42, 'train_batch_size': 256,
          'infer_batch_size': 96, 'beam_size': 30, 'train': {}, 'eval': {},
          'checkpoint_modified': False, 'compile_requested': args.compile}
report_path = OUT / ('probe-buckets-compile.json' if args.buckets and args.compile else
                     'probe-buckets.json' if args.buckets else
                     'probe-compile.json' if args.compile else 'probe.json')

def save(phase=None):
    if phase:
        report['phase'] = phase
    p = report_path.with_suffix('.tmp')
    p.write_text(json.dumps(report, indent=2) + '\n')
    p.replace(report_path)
    print(json.dumps({'phase': report['phase'], 'train': report['train'],
                      'eval': report['eval']}, ensure_ascii=True), flush=True)

def binding(key):
    return gin.query_parameter('train.' + key)

config = {k: binding(k) for k in ['num_layers', 'num_decoder_layers', 'd_model', 'd_ff',
                                 'num_heads', 'd_kv', 'dropout_rate', 'feed_forward_proj', 'sem_id_dim']}
config.update(vocab_size=769, pad_token_id=0, eos_token_id=0)
state = torch.load(REPO / 'out/tiger/amazon/beauty/best_model.pt', map_location='cpu', weights_only=True)
save()
datasets = {s: AmazonSeqDataset(root='dataset/amazon', train_test_split=s, max_seq_len=50,
                               pretrained_rqvae_path=binding('pretrained_rqvae_path'))
            for s in ['train', 'valid']}
rng = random.Random(42)
train_batches = [t5_collate([datasets['train'][i] for i in rng.sample(range(len(datasets['train'])), 256)],
                            256, 3, 150) for _ in range(4)]
if args.buckets:
    order = list(range(len(datasets['train'])))
    rng.shuffle(order)
    order = order[:len(order) // 256 * 256]
    order.sort(key=lambda i: min(len(datasets['train'].samples[i]['history']), 50))
    groups = [order[i:i+256] for i in range(0, len(order), 256)]
    train_batches = []
    for q in range(12):
        group = groups[int((q + .5) * len(groups) / 12)]
        samples = [datasets['train'][i] for i in group]
        length = min(150, max(len(s.item_ids) for s in samples))
        train_batches.append(t5_collate(samples, 256, 3, length))
    report['bucket_probe_lengths'] = [b['input_ids'].shape[1] for b in train_batches]
valid_batches = [t5_collate([datasets['valid'][i] for i in range(start, start + 96)], 256, 3, 150)
                 for start in range(0, 96 * 8, 96)]
report['training_examples'] = len(datasets['train'])
report['validation_users_probed'] = 96 * len(valid_batches)
report['sample_padding_fraction'] = 1 - sum(b['attention_mask'].sum().item() for b in train_batches) / (4 * 256 * 150)
report['sample_padding_fraction'] = 1 - sum(b['attention_mask'].sum().item() for b in train_batches) / sum(b['input_ids'].numel() for b in train_batches)
gpu_batches = [{k: b[k].cuda() for k in ['input_ids', 'attention_mask', 'labels']} for b in train_batches]
save('training_probe')

def amp_context(enabled):
    return torch.autocast('cuda', dtype=torch.float16, enabled=enabled)

modes = [('fp32', False, False), ('fp16', True, False)]
if args.compile:
    modes = [('fp32_compiled', False, True), ('fp16_compiled', True, True)]
for name, amp, compiled in modes:
    set_seed(42, deterministic=True)
    model = Tiger(config).cuda()
    model.load_state_dict(state)
    model.train()
    forward = torch.compile(model.forward, mode='default', dynamic=args.buckets) if compiled else model.forward
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=1024)
    skipped = 0
    losses = []
    save(name + '_warmup')
    warm_started = time.monotonic()

    def step(batch):
        global skipped
        opt.zero_grad()
        with amp_context(amp):
            loss, _ = forward(**batch)
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < old_scale:
            skipped += 1
        value = loss.item()
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError(name + ': nonfinite loss')
        return value

    for i in range(8):
        step(gpu_batches[i % len(gpu_batches)])
    torch.cuda.synchronize()
    warm_seconds = time.monotonic() - warm_started
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for i in range(args.steps):
        losses.append(step(gpu_batches[i % len(gpu_batches)]))
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    opt.zero_grad()
    with amp_context(amp):
        loss, _ = forward(**gpu_batches[0])
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    report['train'][name] = {'steps': args.steps, 'seconds': elapsed,
        'milliseconds_per_step': elapsed * 1000 / args.steps, 'examples_per_second': args.steps * 256 / elapsed,
        'warmup_compile_seconds': warm_seconds, 'finite_gradients': finite,
        'loss_first': losses[0], 'loss_last': losses[-1], 'skipped_updates': skipped,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    if not finite:
        raise RuntimeError(name + ': nonfinite unscaled gradients')
    save(name + '_done')
    del forward, opt, scaler, model, loss
    gc.collect()
    torch.cuda.empty_cache()

if not args.compile:
    model = Tiger(config).cuda()
    model.load_state_dict(state)
    model.eval()
    model_class = type(model.model)
    original_reorder = model_class._reorder_cache

    def reorder_self_only(past_key_values, beam_idx):
        # Cross-attention KV is identical for every beam belonging to one user;
        # beam_idx never moves across users. Keep this cache instead of copying it.
        return tuple((tuple(x.index_select(0, beam_idx) for x in layer[0]),) + tuple(layer[1:])
                     for layer in past_key_values)

    reference = None
    eval_modes = [('fp32', False, False), ('fp16', True, False),
                  ('fp32_cache', False, True), ('fp16_cache', True, True)]
    if args.buckets:
        eval_modes = [('fp32', False, False), ('fp32_bucket_cache', False, True),
                      ('fp16_bucket_cache', True, True)]
    for name, amp, fast_cache in eval_modes:
        model_class._reorder_cache = staticmethod(reorder_self_only if fast_cache else original_reorder)
        save(name + '_evaluation')
        predictions = []
        metrics = {f'{m}@{k}': [] for m in ['Recall', 'NDCG'] for k in [5, 10]}
        batches = valid_batches
        restore_order = None
        if 'bucket' in name:
            restore_order = sorted(range(768), key=lambda i: min(len(datasets['valid'].samples[i]['history']), 50))
            batches = []
            for start_index in range(0, 768, 96):
                samples = [datasets['valid'][i] for i in restore_order[start_index:start_index+96]]
                length = min(150, max(len(s.item_ids) for s in samples))
                batches.append(t5_collate(samples, 256, 3, length))
        with torch.no_grad(), amp_context(amp):
            b = batches[0]
            model.generate(b['input_ids'].cuda(), b['attention_mask'].cuda(), num_beams=30)
            torch.cuda.synchronize()
            start = time.monotonic()
            for b in batches:
                pred = model.generate(b['input_ids'].cuda(), b['attention_mask'].cuda(), num_beams=30)
                pred = pred[:, 1:].reshape(96, 30, 3).cpu()
                predictions.append(pred)
                pos = calculate_pos_index(pred, b['labels'], maxk=30)
                for k in [5, 10]:
                    metrics[f'Recall@{k}'].append(recall_at_k(pos, k))
                    metrics[f'NDCG@{k}'].append(ndcg_at_k(pos, k))
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
        pred = torch.cat(predictions)
        if restore_order is not None:
            restored = torch.empty_like(pred)
            restored[torch.tensor(restore_order)] = pred
            pred = restored
            metrics = {f'{m}@{k}': [] for m in ['Recall', 'NDCG'] for k in [5, 10]}
            for bi, b in enumerate(valid_batches):
                pos = calculate_pos_index(pred[bi*96:(bi+1)*96], b['labels'], maxk=30)
                for k in [5, 10]:
                    metrics[f'Recall@{k}'].append(recall_at_k(pos, k))
                    metrics[f'NDCG@{k}'].append(ndcg_at_k(pos, k))
        if reference is None:
            reference = pred
        identical = torch.equal(pred, reference)
        report['eval'][name] = {'seconds': elapsed, 'users_per_second': len(pred) / elapsed,
            'exact_same_ranked_predictions_as_fp32': identical,
            'same_top10_slots_fraction': (pred[:, :10] == reference[:, :10]).all(-1).float().mean().item(),
            'metrics': {k: sum(v) / len(v) for k, v in metrics.items()}}
        if name == 'fp32_cache' and not identical:
            raise RuntimeError('Cross-cache optimization changed FP32 generated predictions')
        save(name + '_done')
    model_class._reorder_cache = staticmethod(original_reorder)
report['state'] = 'complete'
save('complete')
