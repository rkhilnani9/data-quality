import os
import random
import pandas as pd
import numpy as np

from datetime import datetime
from collections import Counter
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)

from data_quality.config import *


def get_sentences_from_data(df, cat_cols, num_cols):
    cols = cat_cols + num_cols

    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].fillna("BLANK")

    cat_df = df[cat_cols]
    num_df = df[num_cols]

    for col in num_cols:
        num_df[col] = num_df[col].rank() // int(num_df.shape[0] / 100)
        num_df[col] = num_df[col].astype(int)
        num_df[col] = num_df[col].astype(str)
    df_final = pd.concat([cat_df, num_df], axis=1)

    df_final["sentences"] = df_final[cols].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    )
    sentences = df_final["sentences"].tolist()

    current_time = datetime.now().strftime("%Y-%m-%d")
    save_path = f"../sentences/{current_time}.txt"

    with open(save_path, "w") as f:
        for item in sentences[:20000]:
            f.write("%s\n" % item)

    return sentences, save_path


def train_tokenizer(sentences, sentences_path):
    word_count = []
    for sent in sentences:
        count = Counter(sent.split(" "))
        word_count.append(count.keys())

    word_count = [item for sublist in word_count for item in sublist]
    vocab_size = len(set(word_count))

    paths = [sentences_path]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(
        files=paths,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>",],
    )

    current_time = datetime.now().strftime("%Y-%m-%d")
    save_path = f"../tokenizers/{current_time}/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tokenizer.save_model(save_path)

    return save_path, vocab_size


def process_data_for_language_modeling(tokenizer_path, sentences_path):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer, file_path=sentences_path, block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    return dataset, data_collator


def train_language_model(tokenizer_path, sentences_path):
    config = RobertaConfig(
        # vocab_size=vocab_size,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
    )

    model = RobertaForMaskedLM(config=config)

    training_args = TrainingArguments(
        num_train_epochs=TRAIN_EPOCHS, per_device_train_batch_size=TRAIN_BATCH_SIZE,
    )

    dataset, data_collator = get_dataset_for_language_modeling(
        tokenizer_path, sentences_path
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    current_time = datetime.now().strftime("%Y-%m-%d")
    save_path = f"../models/{current_time}/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainer.save_model(save_path)

    return save_path


def get_fill_mask_pipeline(tokenizer_path, model_path):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    fill_mask = pipeline("fill-mask", model=model_path, tokenizer=tokenizer)

    return fill_mask


def prepare_labels(df):
    df2 = (
        df.set_index(["sentence", "masked_sentence", "masked_index"])
        .apply(pd.Series.explode)
        .reset_index()
    )
    df2["score_filter"] = np.where(df2["scores"] > 0.2, 1, 0)

    labels = []
    for idx, row in df2.iterrows():
        if (row["predicted_tokens"] == row["sentence"].split()[row["masked_index"]]) & (
            row["score_filter"] == 1
        ):
            labels.append(1)
        else:
            labels.append(0)

    df2["label"] = label

    final_df = df2[["predicted_sentence", "label", "masked_index"]]
    return final_df


def fill_mask_prediction(tokenizer_path, model_path, sentences):
    masked_sentences = []
    masked_indices = []
    for sent in sentences[:NUM_ROWS_FOR_FILL_MASK]:
        words = sent.split(" ")
        idx_to_mask = random.randint(0, len(words) - 1)
        words[idx_to_mask] = "<mask>"
        masked_sent = " ".join(words)
        masked_sentences.append(masked_sent)
        masked_indices.append(idx_to_mask)

    fill_mask = get_fill_mask_pipeline(tokenizer_path, model_path)

    fill_mask_predictions = fill_mask(masked_sentences)

    scores = []
    predicted_sents = []
    predicted_tokens = []
    for pred in fill_mask_predictions:
        scores.append([p["score"] for p in pred])
        predicted_sents.append(
            [p["sequence"].lstrip("<s>").rstrip("</s>").strip(" ") for p in pred]
        )
        predicted_tokens.append(
            [tokenizer.convert_ids_to_tokens(p["token"]).strip("Ġ") for p in pred]
        )

    df = pd.DataFrame()
    df["sentence"] = sentences_new
    df["masked_sentence"] = masked_sentences
    df["masked_index"] = masked_indices
    df["predicted_sentence"] = predicted_sents
    df["scores"] = scores
    df["predicted_tokens"] = predicted_tokens

    final_df = prepare_labels(df)

    current_time = datetime.now().strftime("%Y-%m-%d")
    save_path = f"../data_for_clf/{current_time}.csv/"

    final_df.to_csv(save_path)

    return save_path
