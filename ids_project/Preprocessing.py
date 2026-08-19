"""
Downloads-agnostic preprocessing + non-IID partitioning for CIC-IDS2017.

Merges the selected daily capture CSVs, cleans them, encodes the label,
standardizes features, partitions the result across clients using a
Dirichlet distribution (non-IID), and saves one train/test tensor pair
per client under `output_dir`.
"""

import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from flwr_datasets.partitioner import DirichletPartitioner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import TensorDataset


def preprocess_and_partition_multiple_files(
    file_paths, num_partitions, alpha=0.1, output_dir="data"
):
    print("Loading and merging input CSV files...")

    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            print(f"UTF-8 decode error on {path}, retrying with 'latin1'...")
            df = pd.read_csv(path, encoding="latin1")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print("Merged dataset shape:", df.shape)

    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if "Label" not in df.columns:
        raise ValueError("Column 'Label' was not found in the dataset.")

    print("Encoding the 'Label' column...")
    label_encoder = OrdinalEncoder()
    df["Label"] = label_encoder.fit_transform(df[["Label"]])

    y = df["Label"]
    X = df.drop("Label", axis=1)

    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        X = X.drop(columns=non_numeric_cols)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    processed_df = pd.concat([X_scaled, y], axis=1)
    print("Preprocessing done. Final shape:", processed_df.shape)

    print(f"Partitioning into {num_partitions} clients (Dirichlet, alpha={alpha})...")
    hf_dataset = HFDataset.from_pandas(processed_df)

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions, alpha=alpha, partition_by="Label"
    )
    partitioner._dataset = hf_dataset

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory '{output_dir}'.")

    print("Saving per-client train/test partitions...")
    for i in range(num_partitions):
        partition = partitioner.load_partition(i)

        y_part = np.array(partition["Label"]).reshape(-1, 1).astype(np.float32)
        X_part_df = partition.remove_columns("Label").to_pandas()
        X_part = X_part_df.values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X_part, y_part, test_size=0.2, random_state=42
        )

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

        torch.save(train_dataset, os.path.join(output_dir, f"train_partition_{i}.pt"))
        torch.save(test_dataset, os.path.join(output_dir, f"test_partition_{i}.pt"))

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    # Subset of CIC-IDS2017 daily capture files used for the experiments.
    # Download the full dataset from:
    # https://www.unb.ca/cic/datasets/ids-2017.html
    FILE_PATHS = [
        "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    ]
    NUM_CLIENTS = 10
    DIRICHLET_ALPHA = 0.5

    preprocess_and_partition_multiple_files(
        file_paths=FILE_PATHS,
        num_partitions=NUM_CLIENTS,
        alpha=DIRICHLET_ALPHA,
        output_dir="data",
    )
