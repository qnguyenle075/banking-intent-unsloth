import argparse
import json
import torch
import yaml
from unsloth import FastLanguageModel


INFERENCE_TEMPLATE = (
    "Below is a customer message sent to a bank's support system. "
    "Classify the message into exactly one intent category. "
    "Output only the intent label, nothing else.\n\n"
    "### Message:\n{message}\n\n"
    "### Intent:\n"
)


class IntentClassification:
    def __init__(self, model_path: str):
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)

        with open(self.config["label_map_path"], "r") as f:
            self.label_map = json.load(f)

        self.valid_labels = set(self.label_map.values())

        checkpoint = self.config["checkpoint_path"]
        print(f"Loading model from {checkpoint}...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=self.config["max_seq_length"],
            load_in_4bit=self.config.get("load_in_4bit", True),
            dtype=None,
        )

        FastLanguageModel.for_inference(self.model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded. {len(self.label_map)} intent classes available.")

    def __call__(self, message: str) -> str:
        prompt = INFERENCE_TEMPLATE.format(message=message.strip())
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        max_new_tokens = self.config.get("max_new_tokens", 16)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][prompt_length:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        predicted_label = decoded.split("\n")[0].strip()

        if predicted_label not in self.valid_labels:
            for label in self.valid_labels:
                if label.startswith(predicted_label) or predicted_label.startswith(label):
                    predicted_label = label
                    break

        return predicted_label


def main():
    parser = argparse.ArgumentParser(description="Banking Intent Inference")
    parser.add_argument(
        "--config", type=str, default="configs/inference.yaml",
        help="Path to inference config YAML",
    )
    parser.add_argument(
        "--message", type=str, default=None,
        help="Customer message to classify",
    )
    args = parser.parse_args()

    classifier = IntentClassification(args.config)

    if args.message:
        label = classifier(args.message)
        print(f"\nMessage: {args.message}")
        print(f"Predicted Intent: {label}")
    else:
        print("\n" + "=" * 60)
        print("Banking Intent Classifier — Interactive Mode")
        print("Type a message and press Enter. Type 'quit' to exit.")
        print("=" * 60)

        while True:
            message = input("\n> Message: ").strip()
            if message.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not message:
                continue
            label = classifier(message)
            print(f"  → Intent: {label}")


if __name__ == "__main__":
    main()
