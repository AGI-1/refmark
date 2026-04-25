import pytest

from refmark_train import backends


def test_detect_backend_accepts_cuda_when_torch_reports_available(monkeypatch):
    monkeypatch.setattr(backends, "_cuda_available", lambda: True)
    monkeypatch.setattr(backends, "_directml_available", lambda: False)

    assert backends.detect_backend("cuda") == "cuda"
    assert backends.detect_backend("auto") == "cuda"


def test_detect_backend_rejects_requested_cuda_when_unavailable(monkeypatch):
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)

    with pytest.raises(ValueError, match="cuda backend requested"):
        backends.detect_backend("cuda")
