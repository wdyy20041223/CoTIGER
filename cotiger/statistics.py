"""Rebuild the fixed Train-only SID transition counts; no model training."""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .evaluate import file_sha256
from .ids import PREFIX_SPACE, RADIX, SID_SPACE, prefix_keys, sid_keys


def train_statistics(path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    before = file_sha256(path)
    targets, previous = [], []
    columns = ['sample_id', 'input_ids', 'attention_mask', 'target_tokens']
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            if not row['sample_id'].startswith('train_') or row['attention_mask'][-3:] != [1, 1, 1]:
                raise ValueError('Expected Train samples with a complete last-history SID')
            targets.append(row['target_tokens'])
            previous.append(row['input_ids'][-3:])
    if len(targets) != 131413:
        raise ValueError(f'Expected the fixed Beauty Train split (131413), found {len(targets)}')
    targets, previous = np.array(targets), np.array(previous)
    data = {'samples': np.array(len(targets))}
    for name, current, prior, space in (
        ('sid', sid_keys(targets), sid_keys(previous), SID_SPACE),
        ('prefix', prefix_keys(targets), prefix_keys(previous), PREFIX_SPACE),
    ):
        for group, values in (('target', current), ('context', prior), ('pair', prior * space + current)):
            keys, counts = np.unique(values, return_counts=True)
            data[f'{name}_{group}_keys'] = keys
            data[f'{name}_{group}_counts'] = counts.astype(np.int32)
    data['code1_counts'] = np.bincount(targets[:, 0], minlength=RADIX).astype(np.int32)
    if before != file_sha256(path):
        raise RuntimeError('Train source changed while reading')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(f'Saved {len(targets)} Train samples; source SHA256: {before}')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('artifacts/beauty/train_samples.parquet'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train_statistics(args.train, args.output)


if __name__ == '__main__':
    main()
