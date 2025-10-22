from __future__ import annotations
import os
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr
import json, sqlite3
from datetime import datetime


router = APIRouter(prefix="/billing", tags=["Billing"])

# ------------------------
# Env / Stripe helpers
# ------------------------
def _env(name: str, alt: str | None = None, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if not v and alt:
        v = os.getenv(alt)
    return v if v is not None else default

def _get_stripe():
    import stripe
    key = _env("STRIPE_SECRET_KEY", alt="STRIPE_SECRET")
    if not key:
        raise HTTPException(status_code=500, detail="Stripe secret key not configured")
    stripe.api_key = key
    return stripe

def _get_price_id() -> str:
    pid = _env("STRIPE_PRICE_ID", alt="PRICE_MONTHLY")
    if not pid:
        raise HTTPException(status_code=500, detail="Stripe price id not configured")
    return pid

def _get_webhook_secret() -> str:
    wh = _env("STRIPE_WEBHOOK_SECRET")
    if not wh:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured")
    return wh

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://127.0.0.1:8501")

# Comma-separated list of free users (emails). Example: FREE_USERS="a@x.com,b@y.com"
FREE_USERS = {e.strip().lower() for e in os.getenv("FREE_USERS", "").split(",") if e.strip()}

def _is_free(email: str | None) -> bool:
    return bool(email) and email.lower() in FREE_USERS

DB_PATH = os.getenv("BILLING_DB_PATH", os.path.join(os.path.dirname(__file__), "tailwing.db"))

def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def _db_init():
    conn = _db()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        stripe_customer_id TEXT,
        subscription_id TEXT,
        price_id TEXT,
        referrer TEXT,
        status TEXT,
        entitlements_json TEXT,
        updated_at TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        name TEXT,
        props_json TEXT,
        ts TEXT
    )""")
    conn.commit()
    conn.close()

_db_init()



# ------------------------
# API models
# ------------------------
class CheckoutRequest(BaseModel):
    email: EmailStr
    # Optional: capture referral handle for attribution in webhooks
    referrer: Optional[str] = None
    # Optional: if you later want to pre-apply a specific promotion_code ID
    # promotion_code: Optional[str] = None


# ------------------------
# Routes
# ------------------------
@router.post("/checkout")
async def create_checkout(data: CheckoutRequest):
    """
    Creates a Stripe Checkout Session for a subscription.
    - Allows promotion codes (user can type a promo in the Checkout UI).
    - Stores optional referrer in metadata for attribution on webhook.
    """
    stripe = _get_stripe()

        # --- Short-circuit for whitelisted free users ---
    if _is_free(str(data.email)):
        user = upsert_user(
            email=str(data.email),
            stripe_customer_id=None,
            subscription_id=None,
            price_id=None,
            referrer=data.referrer,
            status="active",
        )
        activate_entitlements(user, _get_price_id())
        print(f"[billing] Free user: {data.email} -> full entitlements granted")
        return {"url": f"{FRONTEND_URL}?sub=free"}

    metadata: Dict[str, Any] = {}
    if data.referrer:
        metadata["referrer"] = data.referrer

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": _get_price_id(), "quantity": 1}],
        customer_email=str(data.email),
        allow_promotion_codes=True,  # lets users enter promo codes at checkout
        metadata=metadata or None,
        success_url=f"{FRONTEND_URL}?sub=success",
        cancel_url=f"{FRONTEND_URL}?sub=cancel",
    )
    return {"url": session.url}


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Verifies the Stripe webhook and handles key lifecycle events.
    Currently logs and calls lightweight stubs for user upsert + entitlement activation.
    """
    stripe = _get_stripe()
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    # Verify signature
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, _get_webhook_secret())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")

    event_type = event.get("type")
    data = event.get("data", {}).get("object", {})  # type: ignore[assignment]

    # ---- Event handling ----
    if event_type == "checkout.session.completed":
        # Safely extract values
        session = data
        email = (session.get("customer_details") or {}).get("email")
        customer_id = session.get("customer")
        referrer = (session.get("metadata") or {}).get("referrer")
        subscription_id = session.get("subscription")

        # Get price_id reliably by retrieving the subscription (line items often not on the session)
        price_id = None
        if subscription_id:
            try:
                sub = stripe.Subscription.retrieve(subscription_id, expand=["items.data.price"])
                items = sub["items"]["data"]
                if items:
                    price_id = items[0]["price"]["id"]
            except Exception as e:
                print(f"[billing] Failed to retrieve subscription items: {e}")

        user = upsert_user(
            email=email,
            stripe_customer_id=customer_id,
            subscription_id=subscription_id,
            price_id=price_id,
            referrer=referrer,
            status="active",  # on successful checkout we treat as active
        )
        activate_entitlements(user, price_id or _get_price_id())
        print(f"[billing] checkout.session.completed email={email} customer={customer_id} price_id={price_id}")

    elif event_type == "customer.subscription.updated":
        customer_id = data.get("customer")
        status = data.get("status")  # trialing | active | past_due | canceled | unpaid | incomplete...
        price_id = None
        try:
            items = (data.get("items") or {}).get("data") or []
            if items:
                price_id = items[0].get("price", {}).get("id")
        except Exception:
            pass

        user = upsert_user(
            email=None,  # may not be present on this event
            stripe_customer_id=customer_id,
            subscription_id=data.get("id"),
            price_id=price_id,
            referrer=None,
            status=status,
        )
        # Update entitlements if price changed or status transitioned
        if status == "active":
            activate_entitlements(user, price_id or _get_price_id())
        elif status in {"canceled", "unpaid", "incomplete_expired"}:
            deactivate_entitlements(user)
        print(f"[billing] customer.subscription.updated customer={customer_id} status={status} price_id={price_id}")

    elif event_type == "customer.subscription.deleted":
        customer_id = data.get("customer")
        user = upsert_user(
            email=None,
            stripe_customer_id=customer_id,
            subscription_id=data.get("id"),
            price_id=None,
            referrer=None,
            status="canceled",
        )
        deactivate_entitlements(user)
        print(f"[billing] customer.subscription.deleted customer={customer_id}")

    elif event_type == "invoice.paid":
        # Renewal succeeded; you can extend access or mark invoice paid (already implied by 'active')
        customer_id = data.get("customer")
        print(f"[billing] invoice.paid customer={customer_id}")

    elif event_type == "invoice.payment_failed":
        # Dunning: consider restricting premium features until resolved
        customer_id = data.get("customer")
        print(f"[billing] invoice.payment_failed customer={customer_id}")

    else:
        # It's fine to ignore unhandled events (still return 200 so Stripe doesn't retry)
        print(f"[billing] Unhandled event type: {event_type}")

    return {"status": "ok"}


@router.get("/config_check")
def billing_config_check():
    return {
        "has_secret_key": bool(_env("STRIPE_SECRET_KEY") or _env("STRIPE_SECRET")),
        "has_price_id": bool(_env("STRIPE_PRICE_ID") or _env("PRICE_MONTHLY")),
        "has_webhook_secret": bool(_env("STRIPE_WEBHOOK_SECRET")),
        "frontend_url": FRONTEND_URL,
    }

class TrackEvent(BaseModel):
    email: Optional[EmailStr] = None  # pass current user email if you have it
    name: str
    props: Optional[dict] = None

@router.post("/track")
async def track(ev: TrackEvent):
    conn = _db()
    conn.execute(
        "INSERT INTO events (user_email, name, props_json, ts) VALUES (?, ?, ?, ?)",
        (str(ev.email) if ev.email else None, ev.name, json.dumps(ev.props or {}), datetime.utcnow().isoformat())
    )
    conn.commit(); conn.close()
    return {"ok": True}



# ------------------------
# Minimal stubs (replace with real DB later)
# ------------------------
def upsert_user(
    email: Optional[str],
    stripe_customer_id: Optional[str],
    subscription_id: Optional[str],
    price_id: Optional[str],
    referrer: Optional[str],
    status: Optional[str],
) -> Dict[str, Any]:
    user = {
        "email": email,
        "stripe_customer_id": stripe_customer_id,
        "subscription_id": subscription_id,
        "price_id": price_id,
        "referrer": referrer,
        "status": status,
    }
    entitlements = []  # set/overwrite below only if we activate
    now = datetime.utcnow().isoformat()

    conn = _db()
    # upsert by email (fallback to stripe_customer_id if email missing)
    key_email = email or f"__noemail__{stripe_customer_id or 'unknown'}"
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (email, stripe_customer_id, subscription_id, price_id, referrer, status, entitlements_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            stripe_customer_id=excluded.stripe_customer_id,
            subscription_id=excluded.subscription_id,
            price_id=excluded.price_id,
            referrer=COALESCE(excluded.referrer, users.referrer),
            status=excluded.status,
            updated_at=excluded.updated_at
    """, (key_email, stripe_customer_id, subscription_id, price_id, referrer, status, json.dumps(entitlements), now))
    conn.commit()
    conn.close()

    user["email"] = key_email
    user["entitlements"] = entitlements
    print(f"[billing] upsert_user(db) -> {user}")
    return user

def _price_to_entitlements(price_id: str) -> list[str]:
    try:
        default_price = _get_price_id()
    except Exception:
        default_price = price_id
    if price_id == default_price:
        return ["nba", "nfl", "mlb"]  # all-access for now
    return ["nba"]

def activate_entitlements(user: Dict[str, Any], price_id: str) -> Dict[str, Any]:
    entitlements = _price_to_entitlements(price_id)
    user["entitlements"] = entitlements
    user["status"] = user.get("status") or "active"

    conn = _db()
    conn.execute("UPDATE users SET entitlements_json=?, status=?, updated_at=? WHERE email=?",
                 (json.dumps(entitlements), user["status"], datetime.utcnow().isoformat(), user["email"]))
    conn.commit(); conn.close()
    print(f"[billing] activate_entitlements(db) -> {entitlements}")
    return user

def deactivate_entitlements(user: Dict[str, Any]) -> Dict[str, Any]:
    user["entitlements"] = []
    conn = _db()
    conn.execute("UPDATE users SET entitlements_json=?, status=?, updated_at=? WHERE email=?",
                 (json.dumps([]), "canceled", datetime.utcnow().isoformat(), user["email"]))
    conn.commit(); conn.close()
    print("[billing] deactivate_entitlements(db) -> []")
    return user
