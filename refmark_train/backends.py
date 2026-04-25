from __future__ import annotations


def detect_backend(preference: str = "auto") -> str:
    preference = preference.lower()
    if preference not in {"auto", "cpu", "cuda", "directml", "torchcpu"}:
        raise ValueError(f"unsupported backend: {preference}")
    if preference in {"cpu", "torchcpu"}:
        return preference
    if preference == "cuda":
        if _cuda_available():
            return "cuda"
        raise ValueError("cuda backend requested, but torch.cuda.is_available() is false")
    if _directml_available():
        return "directml"
    if _cuda_available():
        return "cuda"
    return "cpu"


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _directml_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_directml  # noqa: F401
        return True
    except Exception:
        return False
