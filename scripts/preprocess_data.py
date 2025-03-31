import os
import pandas as pd
from datasets import load_dataset


def assign_label(example):
    example["label"] = 1 if example["avg_score"] > 0 else 0
    return example


def preprocess_and_save():
    os.makedirs("data", exist_ok=True)
    dataset = load_dataset("osyvokon/pavlick-formality-scores")

    for split in ["train", "test"]:
        print("Processing split: ", split)
        split_ds = dataset[split].map(assign_label)

        df = pd.DataFrame(split_ds)
        df = df[["sentence", "label"]]

        out_path = os.path.join("data", f"{split}.csv")
        df.to_csv(out_path, index=False)
        print("Saved ", split, "data to ", out_path)


if __name__ == "__main__":
    preprocess_and_save()
