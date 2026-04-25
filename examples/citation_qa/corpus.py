TAX_RATE = 0.075


def normalize_customer_name(name: str) -> str:
    return " ".join(name.strip().title().split())


def compute_invoice_total(subtotal: float, expedited: bool = False) -> float:
    shipping = 18.0 if expedited else 6.0
    taxed = subtotal * (1.0 + TAX_RATE)
    return round(taxed + shipping, 2)


def refund_window_days(customer_tier: str) -> int:
    if customer_tier.lower() == "enterprise":
        return 45
    return 30
