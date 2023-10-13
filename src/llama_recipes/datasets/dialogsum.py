import datasets
from tqdm import tqdm
from itertools import chain

import torch

def dialogsum(tokenizer, dataset_config, split):
    chunk_size=2048
    class Concatenator(object):
        def __init__(self):
            self.residual = {"input_ids": [], "attention_mask": []}
        
        def __call__(self, batch):
            concatenated_samples = {
                k: v + list(chain(*batch[k])) for k, v in self.residual.items()
            }
            total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])
            if total_length >= chunk_size:
                chunk_num = total_length // chunk_size
                result = {
                    k: [
                        v[i : i + chunk_size]
                        for i in range(0, chunk_num * chunk_size, chunk_size)
                    ]
                    for k, v in concatenated_samples.items()
                }
                self.residual = {
                    k: v[(chunk_num * chunk_size) :]
                    for k, v in concatenated_samples.items()
                }
            else:
                result = concatenated_samples
                self.residual = {k: [] for k in concatenated_samples.keys()}
            result["labels"] = result["input_ids"].copy()
            return result

    dataset = datasets.load_dataset(dataset_config, split=split)
    print(f"Type of dataset: {type(dataset[0])}")
    print(f"Keys of dataset: {dataset[0].keys()}")
    print(f"Dataset features: {dataset.features}")
    print(f"Dataset info: {dataset.info}")
    key_counts = {}
    for item in dataset:
        item_keys = item.keys()
        for key in item_keys:
            if key in key_counts:
                key_counts[key] += 1
            else:
                key_counts[key] = 1
    for key, count in key_counts.items():
        print(f"Key: {key}, Count: {count}")



    prompt = (
        f"Summarize this conversation:\n{{dialogue}}\n---\nSummary:\n{{summary}}\n---\Topic:\n{{topic}}{{eos_token}}"
    )


    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                topic=sample["topic"],
                dialogue=sample["dialogue"],
                summary=sample["summary"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    print(f"Prompt applied: {dataset[0]}")
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    print("Concatenator Done!")
    return dataset
