TAX_RATE = 0.0725
LOYALTY_DISCOUNT = 0.12


def apply_tax(amount: float) -> float:
    return round(amount * (1 + TAX_RATE), 2)


def apply_discount(amount: float, code: str | None) -> float:
    """Apply a discount code. SAVE10 takes 10% off; LOYAL takes LOYALTY_DISCOUNT off."""
    if code == "SAVE10":
        return round(amount * 0.9, 2)
    if code == "LOYAL":
        # BUG: subtracts the rate itself instead of a percentage of the amount
        return round(amount - LOYALTY_DISCOUNT, 2)
    return amount


def shipping(subtotal: float) -> float:
    from shopcart.config import FREE_SHIPPING_THRESHOLD, SHIPPING_FEE

    return 0.0 if subtotal >= FREE_SHIPPING_THRESHOLD else SHIPPING_FEE
