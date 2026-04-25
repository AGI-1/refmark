def normalize_name(name: str) -> str:
    return name.strip()


def greeting(name: str) -> str:
    clean = normalize_name(name)
    return f"Hello, {clean}"


def parting(name: str) -> str:
    clean = normalize_name(name)
    return f"Goodbye, {clean}"
