from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))  # load .env BEFORE router imports

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from jose import jwt, JWTError
from billing import router as billing_router
import os

# ---- simple settings (inline for step 1) ----
JWT_SECRET = "change-me-to-a-long-random-string"
JWT_ISSUER = "tailwing"
TOKEN_TTL_MIN = 120

# In-memory “subscriptions” store for step 1 (replace later with DB/Stripe)
# Key = email (lowercased), Value = bool is_subscriber
SUBSCRIPTIONS = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(billing_router)

def mint_session(email: str) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": email.lower().strip(),
        "iss": JWT_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=TOKEN_TTL_MIN)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def parse_session(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"require_exp": True})
        if payload.get("iss") != JWT_ISSUER:
            raise HTTPException(status_code=401, detail="Invalid issuer")
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/auth/dev_login")
def dev_login(payload: dict):
    email = (payload.get("email") or "").lower().strip()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")
    # Create a user session token
    token = mint_session(email)
    # For demo: make anyone with email ending in "+sub@..." a subscriber
    # e.g., chris+sub@example.com -> subscriber=true
    SUBSCRIPTIONS.setdefault(email, email.find("+sub@") != -1)
    return {"session_token": token, "is_subscriber": SUBSCRIPTIONS[email]}

@app.get("/entitlement")
def entitlement(authorization: str | None = Header(None)):
    email = parse_session(authorization)
    is_sub = bool(SUBSCRIPTIONS.get(email, False))
    return {"email": email, "is_subscriber": is_sub}

# Optional helper to toggle subscription for testing
@app.post("/dev/toggle_subscription")
def toggle_sub(payload: dict, authorization: str | None = Header(None)):
    email = parse_session(authorization)
    flag = bool(payload.get("is_subscriber", False))
    SUBSCRIPTIONS[email] = flag
    return {"email": email, "is_subscriber": flag}
