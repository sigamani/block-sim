# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import random
import warnings
from typing import Tuple

import pandas as pd
from datasets import Dataset, Value
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import utils

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression-model-name",
                        type=str,
                        default="roberta-base")
    parser.add_argument("--tokenizer",
                        type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--train-data-path",
                        type=str,
                        default="data/sharegpt-llama-7b-train-40k.json")
    parser.add_argument("--val-data-path",
                        type=str,
                        default="data/sharegpt-llama-7b-val-10k.json")
    parser.add_argument("--output-dir",
                        type=str,
                        default="./model/roberta-length-prediction/llama-2-7b")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-eval-examples", type=int, default=1000)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()
    return args


def generate_regression_dataframe(tokenizer_model,
                                  raw_data,
                                  num_sampled: int = -1) -> Tuple[pd.DataFrame, list]:
    regression_dataset = []
    for i in range(len(raw_data)):
        new_data = []
        new_data.append(raw_data[i]["conversations"][0]["value"])
        len_to_predict = len(
            tokenizer_model.tokenize(raw_data[i]["conversations"][1]["value"]))
        new_data.append(len_to_predict)
        regression_dataset.append(new_data)
    if num_sampled > 0:
        regression_dataset = random.sample(regression_dataset, num_sampled)
    regression_df = pd.DataFrame(regression_dataset, columns=["text", "labels"])
    return regression_df, regression_dataset


def build_dataset(df: pd.DataFrame, tokenizer,
                  max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(df, preserve_index=False)
    max_len = min(max_length, tokenizer.model_max_length)

    def preprocess(batch):
        tokenized = tokenizer(batch["text"],
                              truncation=True,
                              max_length=max_len)
        tokenized["labels"] = [float(x) for x in batch["labels"]]
        return tokenized

    dataset = dataset.map(preprocess,
                           batched=True,
                           remove_columns=["text", "labels"])
    return dataset.cast_column("labels", Value("float32"))


if __name__ == "__main__":
    args = parse_args()
    train_data = utils.jload(args.train_data_path)
    val_data = utils.jload(args.val_data_path)

    length_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                     use_fast=True)
    train_df, _ = generate_regression_dataframe(length_tokenizer, train_data)
    num_eval_examples = min(args.num_eval_examples, len(val_data))
    val_df, val_data_list = generate_regression_dataframe(length_tokenizer,
                                                          val_data,
                                                          num_eval_examples)

    model_tokenizer = AutoTokenizer.from_pretrained(args.regression_model_name,
                                                    use_fast=True)
    if model_tokenizer.pad_token is None and model_tokenizer.eos_token is not None:
        model_tokenizer.add_special_tokens({"pad_token": model_tokenizer.eos_token})

    train_dataset = build_dataset(train_df, model_tokenizer, args.max_seq_length)
    val_dataset = build_dataset(val_df, model_tokenizer, args.max_seq_length)

    data_collator = DataCollatorWithPadding(tokenizer=model_tokenizer,
                                            padding="longest")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.regression_model_name, num_labels=1)
    if model_tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.resize_token_embeddings(len(model_tokenizer))
        model.config.pad_token_id = model_tokenizer.pad_token_id
    model.config.problem_type = "regression"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=10,
        learning_rate=1e-5,
        report_to=[],
    )

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      tokenizer=model_tokenizer,
                      data_collator=data_collator)

    trainer.train()
    trainer.save_model(args.output_dir)
    model_tokenizer.save_pretrained(args.output_dir)
    print("train finished")

    predictions = trainer.predict(val_dataset)
    model_outputs = predictions.predictions.squeeze()
    if model_outputs.ndim == 0:
        model_outputs = [float(model_outputs)]
    else:
        model_outputs = model_outputs.tolist()

    d_max = []
    for i in range(len(val_data_list)):
        x = val_data_list[i][1]
        y = float(model_outputs[i])
        d_max.append(abs(x - y))

    diff = sum(d_max)
    acc_50t = sum(1 if x <= 50 else 0 for x in d_max)
    acc_100t = sum(1 if x <= 100 else 0 for x in d_max)
    samples = len(val_data_list)

    print(f"# Samples: {samples}")
    print(f"Error: {diff / samples if samples else 0.0}")
    print(f"Acc-50: {acc_50t / samples if samples else 0.0}")
    print(f"Acc-100: {acc_100t / samples if samples else 0.0}")
