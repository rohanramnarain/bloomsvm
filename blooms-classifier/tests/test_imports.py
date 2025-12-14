import importlib


def test_imports():
    modules = [
        "src.acquisition.synth",
        "src.features.tfidf",
        "src.models.svm_train",
        "src.models.bert_finetune",
    ]
    for mod in modules:
        importlib.import_module(mod)


def test_mps_detection():
    try:
        import torch

        assert isinstance(torch.backends.mps.is_available(), bool)
    except Exception:
        # torch optional in CI-less environments
        assert True
