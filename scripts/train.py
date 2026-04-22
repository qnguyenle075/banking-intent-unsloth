import argparse
import json
import os
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel


INSTRUCTION_TEMPLATE = (
    "Below is a customer message sent to a bank's support system. "
    "Classify the message into the correct intent category.\n\n"
    "### Message:\n{message}\n\n"
    "### Intent:\n"
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(path: str) -> Dataset:
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)


def formatting_func(examples):
    return examples["text"]


def evaluate_model(model, tokenizer, test_path: str, label_map: dict, max_new_tokens: int = 32):
    FastLanguageModel.for_inference(model)

    df = pd.read_csv(test_path)
    true_labels = []
    pred_labels = []
    correct = 0
    total = len(df)

    # Reverse label map: name -> id
    name_to_id = {v: int(k) for k, v in label_map.items()}
    valid_labels = set(label_map.values())

    print(f"\n  Evaluating on {total} test samples...")

    for idx, row in df.iterrows():
        original_text = row["original_text"]
        true_label_name = row["label_name"]
        true_labels.append(true_label_name)

        prompt = INSTRUCTION_TEMPLATE.format(message=original_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.01,
                do_sample=False,
                use_cache=True,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = decoded.split("### Intent:\n")[-1].strip()
        predicted = predicted.split("\n")[0].strip()

        pred_labels.append(predicted)
        if predicted == true_label_name:
            correct += 1

        if (idx + 1) % 100 == 0:
            print(f"    Progress: {idx + 1}/{total} (acc so far: {correct/(idx+1):.4f})")

    accuracy = correct / total
    print(f"\n  Final Accuracy: {accuracy:.4f} ({correct}/{total})")

    unique_labels = sorted(set(true_labels + pred_labels))
    report = classification_report(
        true_labels, pred_labels, labels=unique_labels, zero_division=0
    )
    print(f"\n  Classification Report:\n{report}")

    return accuracy, report


def main():
    parser = argparse.ArgumentParser(description="Train Banking Intent Classifier")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("Banking Intent Classification — Training with Unsloth")
    print("=" * 60)

    print(f"\n[1/5] Loading model: {config['model_name']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        dtype=None,
    )

    print(f"\n[2/5] Configuring LoRA (r={config['lora_r']}, alpha={config['lora_alpha']})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )

    print(f"\n[3/5] Loading training data from {config['train_data_path']}...")
    train_dataset = load_data(config["train_data_path"])
    print(f"  Train samples: {len(train_dataset)}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        optim=config["optimizer"],
        lr_scheduler_type=config["lr_scheduler_type"],
        seed=config["seed"],
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", False),
        logging_steps=config["logging_steps"],
        save_strategy="epoch",
        save_total_limit=config.get("save_total_limit", 2),
        report_to="none",
    )

    print(f"\n[4/5] Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        formatting_func=formatting_func,
        max_seq_length=config["max_seq_length"],
        packing=False,
    )

    trainer.train()

    print(f"\n  Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n[5/5] Evaluating on test set...")
    label_map_path = config.get("label_map_path", "configs/label_map.json")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    accuracy, report = evaluate_model(
        model, tokenizer, config["test_data_path"], label_map
    )

    results = {"accuracy": accuracy, "report": report}
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  Test Accuracy:  {accuracy:.4f}")
    print(f"  Results:        {results_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
