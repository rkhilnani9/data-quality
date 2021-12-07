import os
import numpy as np
import pandas as pd

from loguru import logger
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from data_quality.config import *


def tokenize(sentences_train, sentences_test, vocab_size, col_count):
    tokenizer = Tokenizer(num_words=vocab_size + 1)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1

    maxlen = col_count + 1

    X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
    X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)

    return X_train, X_test


def classifier(vocab_size, col_count):
    embedding_dim = EMBEDDING_DIM
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model = Sequential()
    model.add(
        layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, input_length=col_count + 1
        )
    )

    model.add(layers.Conv1D(embedding_dim, 5, activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(HIDDEN_DIM, activation="relu"))
    model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.Dense(col_count + 1, activation="sigmoid"))
    model.compile(optimizer=opt, loss="binary_crossentropy")
    model.summary()


def prepare_data_for_classsifier(df, vocab_size, col_count):
    sentences = df["predicted_sentence"].values
    masked_indices = df["masked_index"].values

    b = np.zeros((masked_indices.size, masked_indices.max() + 1))
    b[np.arange(masked_indices.size), masked_indices] = 1

    y = df["label"].values
    y = np.concatenate((y.reshape(-1, 1), b), axis=1)

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.2, random_state=1000
    )

    x_train, x_test = tokenize(sentences_train, sentences_test, vocab_size, col_count)

    return x_train, x_test, y_train, y_test


def train_classifier(classifier_data_path, vocab_size, col_count):
    df = pd.read_csv(classifier_data_path, vocab_size, col_count)

    x_train, x_test, y_train, t_test = prepare_data_for_classsifier(
        df, vocab_size, col_count
    )
    _ = model.fit(
        x_train,
        y_train,
        epochs=CLF_TRAIN_EPOCHS,
        verbose=True,
        validation_data=(x_test, y_test),
        batch_size=CLF_TRAIN_BATCH_SIZE,
    )
