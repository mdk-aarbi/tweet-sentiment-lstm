"""Predict tweet sentiment with the trained BiLSTM + GloVe model.

Usage: python predict.py "your tweet here"
"""
import json, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

NAMES = ["negative", "neutral", "positive"]
vocab = json.load(open("models/vocab.json"))
vec = layers.TextVectorization(max_tokens=20000, output_sequence_length=50, vocabulary=vocab)
model = tf.keras.models.load_model("models/model_B_glove.keras")

text = " ".join(sys.argv[1:]) or "I love this so much!"
p = model.predict(vec(tf.constant([text])), verbose=0)[0]
for n, v in sorted(zip(NAMES, p), key=lambda t: -t[1]):
    print("%-9s %.3f" % (n, v))
