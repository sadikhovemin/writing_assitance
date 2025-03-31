import pandas as pd
from transformers import pipeline


def load_data(csv_path, n_per_class=50):
    df = pd.read_csv(csv_path)
    synthetic_input = []

    for label in [0, 1]:
        subset = df[df["label"] == label]
        sampled = subset.sample(n=min(n_per_class, len(subset)))
        synthetic_input.extend(sampled.to_dict("records"))
    return synthetic_input


def generate_synthetic_examples(examples):
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    synthetic_examples = []
    for ex in examples:
        original = ex["sentence"]
        label = ex["label"]

        if label == 0:
            # original is informal - generating formal variant
            prompt = f'Rewrite this sentence to be much more formal in tone and structure: "{original}"'
            target_label = 1
        else:
            # original is formal - generating informal variant
            prompt = f'Rewrite this sentence using a casual and relaxed tone similar to daily conversations: "{original}"'
            target_label = 0

        messages = [
            {
                "role": "system",
                "content": "Only provide the output conversion response. Please don't add any notes or comments",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            prompt = pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            result = pipe(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            generated = result[0]["generated_text"].split("<|assistant|>")[-1].strip()
        except Exception as e:
            print("Error generating synthetic text: ", e)
            generated = original

        synthetic_examples.append(
            {
                "original_sentence": original,
                "sentence": generated,
                "label": target_label,
            }
        )
    return synthetic_examples


def main():
    examples = load_data(csv_path="data/train.csv", n_per_class=50)
    print("Generating synthetic examples for ", len(examples), " sentences")
    synthetic_data = generate_synthetic_examples(examples)

    df = pd.DataFrame(synthetic_data)
    out_path = "data/synthetic.csv"
    df.to_csv(out_path, index=False)
    print("Saved synthetic examples to ", out_path)


if __name__ == "__main__":
    main()
