import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=64):
        with open(file_path, encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.lines[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze()
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits est numpy
    logits = torch.tensor(logits)   # convertir en tensor CPU
    labels = torch.tensor(labels)   # convertir en tensor CPU

    mask = labels != -100
    true_labels = labels[mask].numpy()
    pred_labels = torch.argmax(logits, dim=-1)[mask].numpy()

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    return {"accuracy": acc, "f1": f1}