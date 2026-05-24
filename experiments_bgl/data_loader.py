import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class HDFSDataset(Dataset):
    """
    DataLoader compatible con el train.py original.
    Aunque se llame HDFSDataset, también sirve para BGL si el CSV ya ha sido
    convertido al formato:
        data/processed/HDFS.log_structured.csv
    con columnas:
        BlockId, EventId
    y etiquetas:
        data/raw/anomaly_label.csv
    con columnas:
        BlockId, Label
    """

    def __init__(self, csv_file, label_file):
        print(f"Carregant CSV processat: {csv_file}")

        try:
            self.logs = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error llegint el CSV: {e}")
            raise e

        print(f"Carregant etiquetes: {label_file}")
        self.labels_df = pd.read_csv(label_file)

        self.label_dict = dict(
            zip(
                self.labels_df["BlockId"].astype(str),
                (self.labels_df["Label"].astype(str).str.lower() == "anomaly").astype(int),
            )
        )

        print("Convertint Events a números...")

        def event_to_int(e):
            try:
                if pd.isna(e):
                    return 1
                e = str(e).strip()
                if e.startswith("E"):
                    return int(e[1:]) + 1
                return int(float(e)) + 1
            except Exception:
                return 1

        self.logs["BlockId"] = self.logs["BlockId"].astype(str)
        self.logs["EventInt"] = self.logs["EventId"].apply(event_to_int).astype(np.int64)

        print("Agrupant seqüències...")
        grouped = self.logs.groupby("BlockId")["EventInt"].apply(list)

        self.data = []
        self.targets = []

        for block_id, events in grouped.items():
            block_id = str(block_id)
            if block_id in self.label_dict:
                seq = np.asarray(events, dtype=np.int64)
                self.data.append(seq)
                self.targets.append(float(self.label_dict[block_id]))

        self.data = np.asarray(self.data, dtype=object)
        self.targets = np.asarray(self.targets, dtype=np.float32)

        print(f"Dades llestes! {len(self.data)} seqüències.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = np.asarray(self.data[idx], dtype=np.int64)
        label = np.float32(self.targets[idx])
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def get_hdfs_data(csv_path, label_path):
    return HDFSDataset(csv_path, label_path)
