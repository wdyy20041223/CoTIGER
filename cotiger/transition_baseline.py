"""The fixed, train-only SID Markov baseline used for the complementarity check."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from .ids import SID_SPACE, lookup, sid_keys


def grouped_rows(contexts, destinations, counts, size):
    keys, inverse = np.unique(contexts * size + destinations, return_inverse=True)
    totals = np.bincount(inverse, weights=counts)
    parents, children = keys // size, keys % size
    boundaries = np.r_[0, np.flatnonzero(parents[1:] != parents[:-1]) + 1, len(parents)]
    return {int(parents[a]): (children[a:b], totals[a:b])
            for a, b in zip(boundaries[:-1], boundaries[1:])}


def topk(scores, tie_order, k=10):
    """Resolve ties by Train popularity, then lexicographic SID, as in the study."""
    ordered = scores[:, tie_order]
    thresholds = np.partition(ordered, ordered.shape[1] - k, axis=1)[:, -k]
    selected = []
    for row, threshold in zip(ordered, thresholds):
        above = np.flatnonzero(row > threshold)
        tied = np.flatnonzero(row == threshold)[:k - len(above)]
        columns = np.r_[above, tied]
        selected.append(tie_order[columns[np.argsort(-row[columns], kind='stable')]])
    return np.asarray(selected, dtype=np.int32)


class SIDMarkov:
    def __init__(self, item_sids, statistics):
        self.item_tokens = item_sids + np.array([1, 257, 513])
        self.keys, inverse, self.group_sizes = np.unique(
            sid_keys(self.item_tokens), return_inverse=True, return_counts=True)
        self.catalog = self.item_tokens[np.unique(inverse, return_index=True)[1]]
        frequency = lookup(statistics['sid_target_keys'], statistics['sid_target_counts'], self.keys)
        self.marginal = (frequency + 1) / (int(statistics['samples']) + len(self.keys))
        self.tie_order = np.lexsort((self.keys, -frequency))
        pairs = statistics['sid_pair_keys']
        self.rows = grouped_rows(pairs // SID_SPACE, np.searchsorted(self.keys, pairs % SID_SPACE),
                                 statistics['sid_pair_counts'], len(self.keys))

    def scores(self, histories):
        values = np.zeros((len(histories), len(self.keys)), np.float32)
        for age in range(3):
            context = histories[:, -age - 1]
            active = context >= 0
            if age:
                active &= (context[:, None] != histories[:, -age:]).all(-1)
            for row in np.flatnonzero(active):
                found = self.rows.get(int(context[row]))
                columns, counts = found if found is not None else (np.array([], dtype=int), np.array([]))
                weight, beta = .5 ** age, 20
                denominator = counts.sum() + beta
                values[row] += weight * beta / denominator * self.marginal
                values[row, columns] += weight * counts / denominator
        for row, history in enumerate(histories):
            seen = np.unique(history[history >= 0])
            columns = np.searchsorted(self.keys, seen)
            values[row, columns[self.group_sizes[columns] == 1]] -= 1000.
        return values


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1 or args.max_samples is not None and args.max_samples < 1:
        raise ValueError('Batch size and sample limit must be positive')
    with np.load(args.artifacts / 'train_statistics.npz', allow_pickle=False) as archive:
        statistics = {key: archive[key] for key in archive.files}
    model = SIDMarkov(np.load(args.artifacts / 'item_semantic_ids.npy', allow_pickle=False), statistics)
    table = pq.read_table(args.artifacts / 'valid_samples.parquet',
                          columns=['sample_index', 'history_item_indices', 'target_sid'])
    if args.max_samples is not None:
        table = table.slice(0, args.max_samples)
    histories = np.full((len(table), 50), -1, np.int64)
    for row, history in enumerate(table['history_item_indices'].to_pylist()):
        history = history[-50:]
        histories[row, -len(history):] = sid_keys(model.item_tokens[history])
    targets = np.array(table['target_sid'].to_pylist()) + [1, 257, 513]
    started = time.perf_counter()
    predictions = np.concatenate([topk(model.scores(histories[i:i+args.batch_size]), model.tie_order)
                                  for i in range(0, len(table), args.batch_size)])
    elapsed = time.perf_counter() - started
    tokens = model.catalog[predictions]
    match = (tokens == targets[:, None]).all(-1)
    ranks = np.where(match.any(1), match.argmax(1) + 1, 0)
    metrics = {}
    for k in (5, 10):
        hit = (ranks > 0) & (ranks <= k)
        metrics[f'Recall@{k}'] = float(hit.mean())
        metrics[f'NDCG@{k}'] = float(np.where(hit, 1 / np.log2(np.maximum(ranks, 1) + 1), 0).mean())
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'predictions.npz', tokens=tokens,
                        sample_ids=np.array([f'valid_{i:07d}' for i in table['sample_index'].to_pylist()]))
    result = dict(method='sid_markov_recent3_sm20', split='valid', samples=len(table),
                  seconds=elapsed, metrics=metrics)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/beauty'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--batch-size', type=int, default=256)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
