"""Tools whose schemas never appear in Whetherman's training data (E3 unseen-schema results).

All are deterministic simulators over small fixtures; schedule_meeting only changes state.
"""

from __future__ import annotations

from whetherman.tools.registry import tool

S = {"type": "string"}
N = {"type": "number"}

PACKAGES = {"halyard-orm": "4.2.0", "bramblekit": "0.19.3", "tidewater": "5.3.2",
            "pikestaff-cli": "1.0.7", "quillstream": "2.11.0"}
RATES_TO_EUR = {"EUR": 1.0, "CAD": 0.66, "USD": 0.92, "GBP": 1.17, "JPY": 0.0061, "CHF": 1.05}
PHRASES = {("good morning", "fr"): "Bonjour", ("thank you", "es"): "Gracias",
           ("shopcart", "fr"): "shopcart", ("tiny shopping-cart library", "fr"):
           "petite bibliothèque de panier d'achat", ("see you tomorrow", "de"): "Bis morgen"}
ORDERS = {"C-1042": [{"order_id": "O-881", "total": 48.25, "status": "shipped"},
                     {"order_id": "O-902", "total": 19.9, "status": "processing"}],
          "C-2001": [{"order_id": "O-915", "total": 120.0, "status": "delivered"}]}


@tool("get_package_version", "Latest published version of a package in the internal registry.",
      {"package": S})
def get_package_version(env, package: str):
    v = PACKAGES.get(package.strip().lower())
    if v is None:
        raise ValueError(f"package {package!r} not found")
    return {"package": package, "latest": v}


@tool("convert_currency", "Convert an amount between currencies at today's rate.",
      {"amount": N, "from_currency": S, "to_currency": S})
def convert_currency(env, amount: float, from_currency: str, to_currency: str):
    f, t = from_currency.strip().upper(), to_currency.strip().upper()
    if f not in RATES_TO_EUR or t not in RATES_TO_EUR:
        raise ValueError("unsupported currency")
    return {"amount": round(float(amount) * RATES_TO_EUR[f] / RATES_TO_EUR[t], 2), "currency": t}


@tool("translate_text", "Translate text into a target language (ISO code, e.g. fr, es, de).",
      {"text": S, "target_lang": S})
def translate_text(env, text: str, target_lang: str):
    key = (text.strip().strip(".!").lower(), target_lang.strip().lower()[:2])
    if key not in PHRASES:
        raise ValueError("translation service has no entry for this text")
    return {"translation": PHRASES[key]}


@tool("query_orders", "List a customer's orders from the order database.", {"customer_id": S})
def query_orders(env, customer_id: str):
    return {"customer_id": customer_id, "orders": ORDERS.get(customer_id.strip().upper(), [])}


@tool("schedule_meeting", "Put a meeting on the shared calendar.",
      {"title": S, "attendees": {"type": "array", "items": S}, "start": S})
def schedule_meeting(env, title: str, attendees: list[str], start: str):
    env.state.setdefault("meetings", []).append({"title": title, "attendees": attendees, "start": start})
    return {"status": "scheduled", "id": f"mtg_{len(env.state['meetings'])}"}


UNSEEN_TOOLS = ["get_package_version", "convert_currency", "translate_text", "query_orders",
                "schedule_meeting"]
