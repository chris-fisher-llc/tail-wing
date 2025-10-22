# app.py — Tail Wing backend (drop-in)
from pathlib import Path
from datetime import datetime, timedelta
import json
import os

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))  # load .env BEFORE other imports

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError

from backend.billing import router as billing_router
from backend.utils_origin import ALLOWED as FILE_ALLOWED, DEFAULT_FRONTEND as FILE_DEFAULT_FRONTEND


# ---------------------------
# Config & Environment
# ---------------------------
# JWT / session
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ISSUER = os.getenv("JWT_ISSUER", "tail-wing-backend")
TOKEN_TTL_MIN = int(os.getenv("TOKEN_TTL_MIN", "10080"))  # default 7 days

# Allowed origins: env wins; fallback to utils_origin
ENV_ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGINS = ENV_ALLOWED if ENV_ALLOWED else list(FILE_ALLOWED)

# Default frontend: env wins; else utils_origin; else first allowed; else empty
DEFAULT_FRONTEND = os.getenv(
    "DEFAULT_FRONTEND_URL",
    FILE_DEFAULT_FRONTEND or (ALLOWED_ORIGINS[0] if ALLOWED_ORIGINS else "")
)

# Optional free-user whitelist preload
SUBSCRIPTIONS: dict[str, bool] = {}  # email -> is_subscriber
_whitelist_path = Path(__file__).with_name("whitelist.json")
if _whitelist_path.exists():
    try:
        data = json.loads(_whitelist_path.read_text(encoding="utf-8"))
        # Accept either a list of emails or {"free":["a@b.com", ...]}
        emails = data if isinstance(data, list) else data.get("free", [])
        for e in emails:
            if isinstance(e, str) and "@" in e:
                SUBSCRIPTIONS[e.lower().strip()] = True
    except Exception:
        # Don’t crash the app if the file is malformed
        pass

# ---------------------------
# FastAPI app & CORS
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Utilities
# ---------------------------
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
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
            options={"require_exp": True}
        )
        if payload.get("iss") != JWT_ISSUER:
            raise HTTPException(status_code=401, detail="Invalid issuer")
        email = (payload.get("sub") or "").lower().strip()
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def pick_frontend_base(request: Request) -> str:
    # 1) trust Origin if it’s allowed (Streamlit sets this)
    origin = (request.headers.get("origin") or "").strip()
    if origin in ALLOWED_ORIGINS:
        return origin
    # 2) optional override via ?origin=
    qp = (request.query_params.get("origin") or "").strip()
    if qp in ALLOWED_ORIGINS:
        return qp
    # 3) fallback
    if DEFAULT_FRONTEND:
        return DEFAULT_FRONTEND
    raise HTTPException(400, "Unknown frontend origin")

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/billing/config_check")
def config_check():
    return {
        "ok": True,
        "allowed": ALLOWED_ORIGINS,
        "default": DEFAULT_FRONTEND,
        "jwt_issuer": JWT_ISSUER,
        "token_ttl_min": TOKEN_TTL_MIN,
    }

# Simple dev sign-in (no email verification)
@app.post("/auth/dev_login")
def dev_login(payload: dict):
    email = (payload.get("email") or "").lower().strip()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")
    token = mint_session(email)

    # Demo heuristic: any email containing "+sub@" is treated as subscriber
    # e.g., chris+sub@example.com -> True
    is_sub = email.find("+sub@") != -1 or SUBSCRIPTIONS.get(email, False)
    SUBSCRIPTIONS[email] = is_sub
    return {"session_token": token, "is_subscriber": is_sub}

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

# Mount billing router (expects it to carry its own prefix)
app.include_router(billing_router)
