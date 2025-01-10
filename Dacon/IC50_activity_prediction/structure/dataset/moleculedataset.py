import torch
from torch.utils.data import Dataset


# SMILES + 분자, 원자 feature dataset
class MoleculeDataset(Dataset):
    def __init__(self, smiles, features, targets=None, tokenizer=None, max_length=512):
        self.smiles = smiles # SMILES
        self.features = features # 분자, 원자 features
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles.iloc[idx]
        features = self.features.iloc[idx].values

        encoding = self.tokenizer.encode_plus(
            smiles,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "features": torch.tensor(features, dtype=torch.float32),
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets.iloc[idx], dtype=torch.float32)

        return item
