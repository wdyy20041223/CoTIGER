"""Original Beauty preprocessing: last 50 items, left padding to multiples of 8."""
import math
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader


def pad_histories(histories):
    width = math.ceil(max(map(len, histories)) / 8) * 8
    ids = np.zeros((len(histories), width), dtype=np.int64)
    masks = np.zeros_like(ids)
    for index, history in enumerate(histories):
        ids[index, -len(history):] = history
        masks[index, -len(history):] = 1
    return torch.from_numpy(ids), torch.from_numpy(masks)



class EvaluationDataset(Dataset):
    def __init__(self, path, item_sids, max_history_items=50, limit=None):
        self.rows = pq.read_table(path, columns=["history_item_indices", "target_sid"]).to_pylist()
        if limit is not None:
            self.rows = self.rows[:limit]
        self.item_sids = item_sids
        self.max_history_items = max_history_items

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        indices = row["history_item_indices"][-self.max_history_items:]
        history = (self.item_sids[np.asarray(indices, dtype=np.int64)] + np.array([1, 257, 513])).reshape(-1).tolist()
        labels = (np.asarray(row["target_sid"], dtype=np.int64) + np.array([1, 257, 513])).tolist()
        return {"history": history, "labels": labels}


def collate_evaluation(rows):
    ids, masks = pad_histories([row["history"] for row in rows])
    return {"input_ids": ids, "attention_mask": masks,
            "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.long)}


def make_loader(dataset, batch_size, collate, shuffle, workers, seed, cuda):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        collate_fn=collate, num_workers=workers, pin_memory=cuda,
        persistent_workers=workers > 0, generator=torch.Generator().manual_seed(seed))



class ValidationDataset(EvaluationDataset):
    def __init__(self, path, item_sids, limit):
        super().__init__(path, item_sids, max_history_items=50, limit=limit)
        indices = pq.read_table(path, columns=["sample_index"])["sample_index"].to_pylist()[:len(self)]
        self.sample_ids = [f"valid_{index:07d}" for index in indices]
        assert len(set(self.sample_ids)) == len(self)

    def __getitem__(self, index):
        row = super().__getitem__(index)
        row["sample_id"] = self.sample_ids[index]
        return row


def collate_validation(rows):
    batch = collate_evaluation(rows)
    batch["sample_ids"] = [row["sample_id"] for row in rows]
    return batch
