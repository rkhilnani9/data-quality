from loguru import logger

from data_quality.lm_utils import (
    get_sentences_from_data,
    train_tokenizer,
    train_language_model,
    fill_mask_prediction,
)
from data_quality.clf_utils import train_classifier


def train(df, cat_cols, num_cols):
    col_count = len(cat_cols + num_cols)
    sentences, sentences_path = get_sentences_from_data(df, cat_cols, num_cols)

    tokenizer_path, vocab_size = train_tokenizer(sentences, sentences_path)

    lm_model_path = train_language_model(tokenizer_path, sentences_path)

    classifier_data_path = fill_mask_prediction(tokenizer_path, lm_model_path, sentences)

    clf_model_path = train_classifier(classifier_data_path, vocab_size, col_count)

    return clf_model_path
