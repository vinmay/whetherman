from shopcart.config import MAX_ITEMS
from shopcart.pricing import apply_discount, apply_tax, shipping


class CartFullError(Exception):
    pass


class Cart:
    def __init__(self) -> None:
        self.items: dict[str, tuple[float, int]] = {}

    def add(self, sku: str, price: float, qty: int = 1) -> None:
        if self.count() + qty > MAX_ITEMS:
            raise CartFullError(f"cart limit is {MAX_ITEMS} items")
        old_price, old_qty = self.items.get(sku, (price, 0))
        self.items[sku] = (price, old_qty + qty)

    def remove(self, sku: str) -> None:
        self.items.pop(sku, None)

    def count(self) -> int:
        return sum(q for _, q in self.items.values())

    def subtotal(self) -> float:
        return round(sum(p * q for p, q in self.items.values()), 2)

    def total(self, code: str | None = None) -> float:
        sub = apply_discount(self.subtotal(), code)
        return round(apply_tax(sub) + shipping(sub), 2)
