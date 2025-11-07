# backend/billing.py
import os
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from jose import jwt, JWTError

# ------------------------
# Config / Environment
# ------------------------
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
PRICE_SINGLE = os.environ.get("PRICE_SINGLE", "")   # optional if client sends price_id
PRICE_ALL = os.environ.get("PRICE_ALL", "")         # optional if client sends price_id
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://127.0.0.1:8501")
DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).resolve().parent.parent / "tailwing.db"))
WHITELIST_PATH = os.environ.get("WHITELIST_PATH", str(Path(__file__).resolve().parent / "whitelist.json"))

if not STRIPE_SECRET_KEY:
    raise RuntimeError("Missing STRIPE_SECRET_KEY")
if not STRIPE_WEBHOOK_SECRET:
    raise RuntimeError("Missing STRIPE_WEBHOOK_SECRET")
if not JWT_SECRET:
    raise RuntimeError("Missing JWT_SECRET")

stripe.api_key = STRIPE_SECRET_KEY

def _mode_from_key(key: str) -> str:
    # Simple but reliable: Stripe keys start with sk_test_ or sk_live_
    return "live" if key.startswith("sk_live_") else "test"

BACKEND_MODE = _mode_from_key(STRIPE_SECRET_KEY)

# ------------------------
# Logging
# ------------------------
logger = logging.getLogger("billing")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
logger.addHandler(handler)

# ------------------------
# Persistence (SQLite)
# ------------------------
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        is_subscriber INTEGER NOT NULL DEFAULT 0,
        plan TEXT,
        stripe_customer_id TEXT,
        stripe_subscription_id TEXT,
        mode TEXT
    );
    """)
    conn.commit()
    return conn

def upsert_user(email: str,
                is_subscriber: bool,
                plan: Optional[str],
                customer_id: Optional[str],
                subscription_id: Optional[str],
                mode: str):
    conn = _db()
    try:
        conn.execute("""
        INSERT INTO users (email, is_subscriber, plan, stripe_customer_id, stripe_subscription_id, mode)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            is_subscriber=excluded.is_subscriber,
            plan=COALESCE(excluded.plan, users.plan),
            stripe_customer_id=COALESCE(excluded.stripe_customer_id, users.stripe_customer_id),
            stripe_subscription_id=COALESCE(excluded.stripe_subscription_id, users.stripe_subscription_id),
            mode=excluded.mode;
        """, (email, 1 if is_subscriber else 0, plan, customer_id, subscription_id, mode))
        conn.commit()
    finally:
        conn.close()

def get_user(email: str) -> Optional[Tuple]:
    conn = _db()
    try:
        cur = conn.execute("SELECT email, is_subscriber, plan, stripe_customer_id, stripe_subscription_id, mode FROM users WHERE email = ?", (email,))
        return cur.fetchone()
    finally:
        conn.close()

# ------------------------
# Whitelist (optional)
# ------------------------
def _load_whitelist() -> set:
    p = Path(WHITELIST_PATH)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return set(e.strip().lower() for e in data)
    except Exception as e:
        logger.warning(f"Failed to read whitelist: {e}")
    return set()

WHITELIST = _load_whitelist()

# ------------------------
# Auth / JWT helpers
# ------------------------
def decode_bearer_token(auth_header: str) -> str:
    """
    Expect `Authorization: Bearer <token>` where token encodes {"email": "..."}.
    """
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        email = payload.get("email")
        if not email:
            raise HTTPException(status_code=401, detail="Token missing email")
        return email.lower()
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ------------------------
# FastAPI Router
# ------------------------
router = APIRouter(prefix="/billing", tags=["billing"])

class CheckoutRequest(BaseModel):
    price_id: Optional[str] = None
    # Optional extra metadata you might want to attach:
    plan: Optional[str] = None  # e.g., "single" or "all"
    # success/cancel override (defaults to FRONTEND_URL with status params)
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

@router.get("/entitlement")
def entitlement(authorization: Optional[str] = Header(default=None)):
    email = decode_bearer_token(authorization)
    # Whitelist wins
    if email in WHITELIST:
        logger.info(f"[ENTITLEMENT] email={email} -> whitelisted -> subscriber=True")
        return {"is_subscriber": True, "email": email, "plan": "whitelist"}
    row = get_user(email)
    if not row:
        logger.info(f"[ENTITLEMENT] email={email} -> not found -> subscriber=False")
        return {"is_subscriber": False, "email": email, "plan": None}
    _, is_sub, plan, _, _, mode = row
    logger.info(f"[ENTITLEMENT] email={email} mode={mode} -> subscriber={bool(is_sub)} plan={plan}")
    return {"is_subscriber": bool(is_sub), "email": email, "plan": plan}

@router.post("/checkout")
def checkout(req: CheckoutRequest, authorization: Optional[str] = Header(default=None)):
    email = decode_bearer_token(authorization)

    # Decide price id
    price_id = req.price_id or PRICE_SINGLE or PRICE_ALL
    if not price_id:
        raise HTTPException(status_code=400, detail="No price_id provided and no default configured")

    success_url = req.success_url or f"{FRONTEND_URL}?status=success"
    cancel_url = req.cancel_url or f"{FRONTEND_URL}?status=cancel"

    # Ensure a Stripe customer per email
    existing_customer_id = None
    row = get_user(email)
    if row and row[3]:  # stripe_customer_id
        existing_customer_id = row[3]

    try:
        if existing_customer_id:
            customer_id = existing_customer_id
        else:
            # Try to find by email first to avoid dupes
            search = stripe.Customer.search(query=f"email:'{email}'")
            if len(search.data) > 0:
                customer_id = search.data[0].id
            else:
                c = stripe.Customer.create(email=email)
                customer_id = c.id

        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={"app_email": email, "requested_plan": (req.plan or "unspecified")},
            subscription_data={
                "metadata": {"app_email": email, "requested_plan": (req.plan or "unspecified")}
            }
        )
        logger.info(f"[CHECKOUT] mode={BACKEND_MODE} email={email} price_id={price_id} session={session.id}")
        # Pre-write/ensure user record exists (not subscriber yet)
        upsert_user(email, is_subscriber=False, plan=req.plan, customer_id=customer_id,
                    subscription_id=None, mode=BACKEND_MODE)

        return {"url": session.url}
    except stripe.error.StripeError as e:
        logger.error(f"[CHECKOUT][STRIPE_ERROR] email={email} {str(e)}")
        raise HTTPException(status_code=502, detail="Stripe error")

@router.post("/webhook")
async def webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload, sig_header=sig, secret=STRIPE_WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        logger.error("[WEBHOOK] Signature verification failed")
        return JSONResponse(status_code=400, content={"ok": False})

    ev_type = event["type"]
    ev_mode = "live" if event.get("livemode") else "test"
    if ev_mode != BACKEND_MODE:
        # Hard guard: don’t flip test events into live (or vice versa)
        logger.warning(f"[WEBHOOK][MODE_MISMATCH] event={ev_type} ev_mode={ev_mode} backend_mode={BACKEND_MODE} -> ignored")
        return JSONResponse(status_code=200, content={"ok": True, "ignored": "mode_mismatch"})

    def _flip(email: Optional[str], is_active: bool, plan: Optional[str], customer_id: Optional[str], sub_id: Optional[str]):
        if not email:
            logger.warning(f"[WEBHOOK][{ev_type}] missing email -> skip flip")
            return
        email_l = email.lower()
        # whitelist stays always-on; do not downgrade it
        if email_l in WHITELIST:
            logger.info(f"[WEBHOOK][{ev_type}] email={email_l} is whitelisted -> force subscriber=True")
            upsert_user(email_l, True, "whitelist", customer_id, sub_id, ev_mode)
            return
        upsert_user(email_l, is_active, plan, customer_id, sub_id, ev_mode)
        logger.info(f"[WEBHOOK][{ev_type}] email={email_l} -> subscriber={is_active} plan={plan} cust={customer_id} sub={sub_id}")

    try:
        # Handle success & ongoing payment signals
        if ev_type == "checkout.session.completed":
            sess = event["data"]["object"]
            email = (sess.get("customer_details") or {}).get("email") or sess.get("customer_email") or (sess.get("metadata") or {}).get("app_email")
            sub_id = sess.get("subscription")
            cust_id = sess.get("customer")
            plan = (sess.get("metadata") or {}).get("requested_plan")
            _flip(email, True, plan, cust_id, sub_id)

        elif ev_type in ("invoice.paid",):
            inv = event["data"]["object"]
            sub_id = inv.get("subscription")
            cust_id = inv.get("customer")
            # Best-effort email retrieval:
            email = None
            try:
                cust = stripe.Customer.retrieve(cust_id) if cust_id else None
                email = (cust or {}).get("email")
            except Exception:
                pass
            plan = (inv.get("lines") or {}).get("data", [{}])[0].get("price", {}).get("product")
            _flip(email, True, plan, cust_id, sub_id)

        # Handle downgrades/cancellations
        elif ev_type in ("customer.subscription.deleted", "invoice.payment_failed"):
            obj = event["data"]["object"]
            sub_id = obj.get("id") if ev_type == "customer.subscription.deleted" else obj.get("subscription")
            cust_id = obj.get("customer")
            email = None
            try:
                cust = stripe.Customer.retrieve(cust_id) if cust_id else None
                email = (cust or {}).get("email")
            except Exception:
                pass
            _flip(email, False, None, cust_id, sub_id)

        else:
            # Other events: log and ignore
            logger.info(f"[WEBHOOK][IGNORED] type={ev_type}")

    except Exception as e:
        logger.exception(f"[WEBHOOK][ERROR] {ev_type}: {e}")
        return JSONResponse(status_code=500, content={"ok": False})

    return JSONResponse(status_code=200, content={"ok": True})
