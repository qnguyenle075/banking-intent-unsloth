import argparse
import json
import os
import random
import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split


PROMPT_TEMPLATE = (
    "Below is a customer message sent to a bank's support system. "
    "Classify the message into exactly one intent category. "
    "Output only the intent label, nothing else.\n\n"
    "### Message:\n{message}\n\n"
    "### Intent:\n{label}"
)

INFERENCE_TEMPLATE = (
    "Below is a customer message sent to a bank's support system. "
    "Classify the message into exactly one intent category. "
    "Output only the intent label, nothing else.\n\n"
    "### Message:\n{message}\n\n"
    "### Intent:\n"
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_intent_names(dataset) -> dict:
    return dict(enumerate(dataset["train"].features["label"].names))


def sample_intents(all_intents: dict, num_intents: int, seed: int) -> list[int]:
    random.seed(seed)
    all_ids = list(all_intents.keys())
    sampled = sorted(random.sample(all_ids, min(num_intents, len(all_ids))))
    return sampled


def preprocess_text(text: str) -> str:
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".?!":
        text += "."
    return text


def format_for_sft(row: dict, intent_names: dict, eos_token: str = "") -> dict:
    message = preprocess_text(row["text"])
    label_name = intent_names[row["label"]]

    formatted_text = PROMPT_TEMPLATE.format(message=message, label=label_name) + eos_token
    return {
        "text": formatted_text,
        "label": row["label"],
        "label_name": label_name,
        "original_text": row["text"],
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess BANKING77 dataset")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("Banking Intent Classification — Data Preprocessing")
    print("=" * 60)

    print("\n[1/5] Loading BANKING77 dataset...")
    dataset = load_dataset(config["dataset_name"], revision="refs/convert/parquet")
    all_intent_names = get_intent_names(dataset)
    print(f"  Total intents: {len(all_intent_names)}")
    print(f"  Total train samples: {len(dataset['train'])}")
    print(f"  Total test samples: {len(dataset['test'])}")

    num_intents = config.get("num_intents", 45)
    seed = config.get("seed", 42)
    print(f"\n[2/5] Sampling {num_intents} intents (seed={seed})...")
    sampled_ids = sample_intents(all_intent_names, num_intents, seed)
    sampled_names = {i: all_intent_names[i] for i in sampled_ids}
    print(f"  Selected intents: {list(sampled_names.values())[:10]}... (showing first 10)")

    new_label_map = {}
    old_to_new = {}
    for new_id, old_id in enumerate(sampled_ids):
        new_label_map[new_id] = all_intent_names[old_id]
        old_to_new[old_id] = new_id

    label_map_path = config.get("label_map_path", "configs/label_map.json")
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    with open(label_map_path, "w") as f:
        json.dump(new_label_map, f, indent=2)
    print(f"  Label map saved to {label_map_path}")

    print(f"\n[3/5] Filtering dataset for selected intents...")
    sampled_id_set = set(sampled_ids)

    all_samples = []
    for split in ["train", "test"]:
        for example in dataset[split]:
            if example["label"] in sampled_id_set:
                all_samples.append(example)

    print(f"  Total filtered samples: {len(all_samples)}")

    print(f"\n[4/5] Formatting data for SFT training...")
    formatted = []
    for sample in all_samples:
        fmt = format_for_sft(sample, all_intent_names)
        fmt["label"] = old_to_new[sample["label"]]
        formatted.append(fmt)

    df = pd.DataFrame(formatted)

    test_size = config.get("test_size", 0.2)
    print(f"\n[5/5] Splitting data (test_size={test_size})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    train_path = config.get("train_data_path", "sample_data/train.csv")
    test_path = config.get("test_data_path", "sample_data/test.csv")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"  Train samples: {len(train_df)} → {train_path}")
    print(f"  Test samples:  {len(test_df)} → {test_path}")
    print(f"  Intents:       {num_intents}")
    print(f"  Label map:     {label_map_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
