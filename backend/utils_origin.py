# utils_origin.py
import os
from fastapi import Request, HTTPException

ALLOWED = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","").split(",") if o.strip()]
DEFAULT_FRONTEND = os.getenv("DEFAULT_FRONTEND_URL", ALLOWED[0] if ALLOWED else "")

def pick_frontend_base(request: Request) -> str:
    origin = request.headers.get("origin", "")
    if origin in ALLOWED:
        return origin
    qp = request.query_params.get("origin")
    if qp in ALLOWED:
        return qp
    if DEFAULT_FRONTEND:
        return DEFAULT_FRONTEND
    raise HTTPException(400, "Unknown frontend origin")
