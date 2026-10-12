"""Audit completed four-dataset results against raw data, logs, and checkpoints.

Run with the reproduction venv only after status.json reports complete.
This reads artifacts and writes results/verification.json; it performs no training.
"""
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

BASE = Path('/root/genrec-reproduction')
REPO = BASE / 'genrec'
SPLITS = ['beauty', 'sports', 'toys', 'home']
KEYS = ['Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10']
COMMIT = 'fb4bd1a1832df0312580a705d3440d1988856e31'

def read_json(path):
    return json.loads(path.read_text())

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def bindings(path):
    text = path.read_text().replace('\\\n', '')
    values = {}
    for key, value in re.findall(r'^([\w.]+)\s*=\s*(.+)$', text, re.MULTILINE):
        try:
            values[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            values[key] = value.strip()
    return values

def require_bindings(path, expected):
    actual = bindings(path)
    for key, value in expected.items():
        assert actual.get(key) == value, (path, key, actual.get(key), value)
    return actual

def main():
    status = read_json(BASE / 'status.json')
    assert status['state'] == 'complete', 'Four-dataset pipeline is still incomplete'
    assert status['completed'] == SPLITS
    assert status['source_commit'] == COMMIT and status['seed'] == 42
    with zipfile.ZipFile(BASE / 'genrec-main.zip') as archive:
        assert archive.comment.decode() == COMMIT
        prefix = archive.namelist()[0].split('/')[0] + '/'
        checked_sources = 0
        for name in archive.namelist():
            relative = name[len(prefix):]
            if not relative or name.endswith('/'):
                continue
            if not (relative.endswith(('.py', '.gin')) or relative in
                    ['requirements.txt', 'pyproject.toml', 'README.md']):
                continue
            expected = archive.read(name)
            if relative == 'genrec/trainers/tiger_trainer.py':
                old = b"with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):"
                new = b"with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False)):"
                assert expected.count(old) == 1
                expected = expected.replace(old, new)
            actual = (REPO / relative).read_text(encoding='utf-8')
            assert actual == expected.decode().replace('\r\n', '\n'), ('Source differs', relative)
            checked_sources += 1
        readme = archive.read(prefix + 'README.md').decode()
    targets = re.findall(r'^\| TIGER \|([^\n]+)', readme, re.MULTILINE)
    assert len(targets) == 4
    targets = {split: [float(value.strip()) for value in row.split('|') if value.strip()]
               for split, row in zip(SPLITS, targets)}
    for filename in ['model-status.json', 'data-status.json']:
        manifest = read_json(BASE / 'manifests' / filename)
        assert manifest['state'] == 'ready'
        for item in manifest['files']:
            path = (BASE / 'models_hub/sentence-t5-xl' / item['name'] if 'name' in item
                    else REPO / 'dataset/amazon/raw' / item['split'] / item['filename'])
            assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256'], path
    rows = {row['dataset']: row for row in csv.DictReader((BASE / 'results/comparison.csv').open())}
    assert set(rows) == set(SPLITS)
    markdown = (BASE / 'results/comparison.md').read_text()
    audits = {}
    import torch
    assert torch.__version__ == '2.6.0+cu124' and torch.cuda.is_available()
    assert not torch.cuda.is_bf16_supported(including_emulation=False)
    for split in SPLITS:
        result = read_json(BASE / 'results' / (split + '.json'))
        assert result['state'] == 'complete' and result['split'] == split
        assert result['seed'] == 42 and result['precision'] == 'fp32'
        rq = require_bindings(BASE / 'manifests' / (split + '-rqvae-resolved.gin'), {
            'train.epochs': 5000, 'train.batch_size': 1024, 'train.learning_rate': .001,
            'train.weight_decay': .0001, 'train.warmup_epochs': 50,
            'train.vae_input_dim': 768, 'train.vae_embed_dim': 32,
            'train.vae_hidden_dims': [512, 256, 128, 64], 'train.vae_n_layers': 3,
            'train.vae_codebook_size': 256, 'train.commitment_weight': .25})
        tiger = require_bindings(BASE / 'manifests' / (split + '-tiger-resolved.gin'), {
            'train.epochs': 200, 'train.batch_size': 256, 'train.infer_batch_size': 96,
            'train.learning_rate': .0001, 'train.weight_decay': 0.0,
            'train.beam_size': 30, 'train.early_stop_patience': 10,
            'train.num_layers': 4, 'train.num_decoder_layers': 4, 'train.d_model': 128,
            'train.d_ff': 1024, 'train.num_heads': 6, 'train.d_kv': 64,
            'train.dropout_rate': .1, 'train.max_seq_len': 50, 'train.sem_id_dim': 3,
            'AmazonSeqDataset.split': split, 'AmazonSeqDataset.add_disambiguation': False})
        checkpoint = REPO / 'out/tiger/amazon' / split / 'rqvae/checkpoint_epoch_4999.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['epoch'] == 4999
        assert all(torch.isfinite(value).all() for value in state['model'].values()
                   if isinstance(value, torch.Tensor) and value.is_floating_point())
        optimizer_steps = sorted({int(value['step']) for value in state['optimizer']['state'].values()
                                  if 'step' in value})
        assert optimizer_steps == [state['scheduler']['last_epoch']]
        assert 'Epoch 5000 Eval' in (BASE / 'logs' / (split + '-rqvae.log')).read_text()
        log = (BASE / 'logs' / (split + '-tiger.log')).read_text().split('Run started:')[-1]
        validations = [(int(epoch), ast.literal_eval(metrics)) for epoch, metrics in
                       re.findall(r'Epoch (\d+) - Valid: (\{[^\n]+\})', log)]
        assert validations
        best_epoch, best_metrics = max(validations, key=lambda item: item[1]['Recall@10'])
        assert best_epoch == result['best_epoch']
        assert abs(best_metrics['Recall@10'] - result['best_valid_recall10_logged']) <= .000050001
        final = re.findall(r'Training done\. Best Valid R@10=[0-9.]+ at epoch (\d+)', log)
        assert final and int(final[-1]) == best_epoch
        final_epoch = max(epoch for epoch, _ in validations)
        assert final_epoch < 200
        if final_epoch < 199:
            assert f'Early stopping at epoch {final_epoch}. Best epoch: {best_epoch}' in log
            assert final_epoch - best_epoch == 10
        best_tests = [ast.literal_eval(metrics) for epoch, metrics in
                      re.findall(r'Epoch (\d+) - Test: (\{[^\n]+\})', log) if int(epoch) == best_epoch]
        assert best_tests
        model_path = REPO / 'out/tiger/amazon' / split / 'best_model.pt'
        assert str(model_path) == result['checkpoint']
        assert digest(model_path) == result['checkpoint_sha256']
        users, items = Counter(), set()
        raw_file = next((REPO / 'dataset/amazon/raw' / split).glob('reviews*.json.gz'))
        with gzip.open(raw_file, 'rt', encoding='utf-8') as stream:
            for line in stream:
                try:
                    review = json.loads(line)
                except json.JSONDecodeError:
                    review = ast.literal_eval(line)
                if review.get('asin') and review.get('reviewerID'):
                    users[review['reviewerID']] += 1
                    items.add(review['asin'])
        assert result['users'] == sum(count >= 5 for count in users.values())
        assert result['items'] == len(items)
        for key, target in zip(KEYS, targets[split]):
            value = result['metrics'][key]
            assert math.isfinite(value) and 0 <= value <= 1
            assert abs(value - best_tests[-1][key]) <= 1e-6, (split, key, 'Final test mismatch')
            assert math.isclose(float(rows[split][key]), value, abs_tol=1e-12)
            assert float(rows[split]['README_' + key]) == target
            assert math.isclose(float(rows[split]['delta_' + key]), value - target, abs_tol=1e-12)
            assert f'{value:.6f}' in markdown
        assert int(rows[split]['best_epoch']) == best_epoch
        audits[split] = {'users': result['users'], 'items': result['items'],
                         'rqvae_epoch': state['epoch'], 'rqvae_optimizer_steps': optimizer_steps,
                         'rqvae_checkpoint_sha256': digest(checkpoint),
                         'best_epoch': best_epoch, 'final_epoch': final_epoch,
                         'best_valid_recall10': best_metrics['Recall@10'],
                         'best_checkpoint_sha256': result['checkpoint_sha256'],
                         'final_test_matches_training_log': True}
    report = {'state': 'verified', 'checked_at': datetime.now(timezone.utc).isoformat(),
              'source_commit': COMMIT, 'checked_source_files': checked_sources,
              'raw_data_and_text_model_hashes_verified': True, 'datasets': audits}
    (BASE / 'results/verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
