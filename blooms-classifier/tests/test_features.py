from src.features.tfidf import build_vectorizer
from src.utils.metrics import threshold_probs
import numpy as np


def test_build_vectorizer():
    word_v, char_v = build_vectorizer(min_df=1)
    sample = ["analyze this question"]
    Xw = word_v.fit_transform(sample)
    Xc = char_v.fit_transform(sample)
    assert Xw.shape[0] == 1
    assert Xc.shape[0] == 1


def test_threshold_probs():
    probs = np.array([[0.2, 0.6]])
    preds = threshold_probs(probs, 0.5)
    assert preds.tolist() == [[0, 1]]
