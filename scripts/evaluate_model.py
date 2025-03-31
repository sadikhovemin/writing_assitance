import os
import argparse
import json
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    texts = df["sentence"].tolist()
    labels = df["label"].tolist()
    return texts, labels


def compute_metrics(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=[0, 1]
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    metrics_dict = {
        "accuracy": acc,
        "precision_informal": prec[0],
        "recall_informal": rec[0],
        "f1_informal": f1[0],
        "precision_formal": prec[1],
        "recall_formal": rec[1],
        "f1_formal": f1[1],
        "confusion_matrix": cm.tolist(),
    }

    return metrics_dict


def evaluate_model_on_texts(model, tokenizer, texts, device="cpu"):
    preds = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu().numpy()[0]
        pred = int(logits.argmax())
        preds.append(pred)

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the test CSV file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on: 'cpu', 'cuda', 'mps'",
    )
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    texts, labels = load_data(args.data)

    model_name = "RoBERTa-base (EN)"
    model_id = "s-nlp/roberta-base-formality-ranker"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(args.device)
    model.eval()

    preds = evaluate_model_on_texts(model, tokenizer, texts, device=args.device)

    metrics_dict = compute_metrics(labels, preds)
    metrics_dict["model"] = model_name

    results = [metrics_dict]

    split_name = os.path.splitext(os.path.basename(args.data))[0]
    results_json = f"results/{split_name}_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation results to {results_json}")


if __name__ == "__main__":
    main()
