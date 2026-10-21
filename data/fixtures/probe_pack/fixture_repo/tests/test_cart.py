import pytest
from shopcart.cart import Cart, CartFullError


def test_add_and_count():
    c = Cart()
    c.add("apple", 1.0, 3)
    c.add("pear", 2.0)
    assert c.count() == 4


def test_subtotal():
    c = Cart()
    c.add("apple", 1.25, 4)
    assert c.subtotal() == 5.0


def test_cart_limit():
    c = Cart()
    with pytest.raises(CartFullError):
        c.add("bulk", 1.0, 41)


def test_save10_discount():
    c = Cart()
    c.add("book", 100.0)
    assert c.total("SAVE10") == round(90 * 1.0725, 2)


def test_loyal_discount():
    c = Cart()
    c.add("book", 100.0)
    assert c.total("LOYAL") == round(88 * 1.0725, 2)


def test_shipping_fee_applies_below_threshold():
    c = Cart()
    c.add("pen", 20.0)
    assert c.total() == 27.95
