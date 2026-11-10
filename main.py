import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)

# Shared secret between the Cloudflare Worker and this backend.
# Only requests that carry it (i.e. coming through the Worker) are served.
GATEWAY_SECRET = os.getenv("GATEWAY_SECRET", "")

# Simple per-visitor limits (kept in memory; resets when the backend restarts).
RATE_PER_MINUTE = int(os.getenv("RATE_PER_MINUTE", "15"))
RATE_PER_DAY = int(os.getenv("RATE_PER_DAY", "200"))

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")
if len(GATEWAY_SECRET) < 20:
    raise RuntimeError(
        "GATEWAY_SECRET is missing or too short in .env (use 20+ random characters)"
    )

app = FastAPI(
    title="Varanasya AI API",
    description="Secure backend for Varanasya AI",
    version="0.4.0"
)

# CORS only matters for browsers calling this backend directly (local testing).
# The Cloudflare Worker calls it server-to-server, so no CORS is needed there.
DEFAULT_ORIGINS = (
    "http://localhost:5500,http://127.0.0.1:5500,"
    "http://localhost:8000,http://127.0.0.1:8000"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
        if o.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- What the frontend is allowed to send on to Gemini -------------------------
ALLOWED_TOP_LEVEL_KEYS = {
    "contents",
    "systemInstruction", "system_instruction",
    "tools",
    "generationConfig", "generation_config",
    "safetySettings", "safety_settings",
}
ALLOWED_TOOL_KEYS = {"google_search", "googleSearch"}
MAX_CONTENT_TURNS = 60

_minute_hits = defaultdict(deque)
_day_hits = defaultdict(deque)


def require_gateway(request: Request):
    given = request.headers.get("x-varanasya-key", "")
    if not secrets.compare_digest(given.encode(), GATEWAY_SECRET.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


def visitor_id(request: Request) -> str:
    # The Worker puts the real visitor IP here (trusted because the secret matched).
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limit(visitor: str):
    now = time.time()
    minute, day = _minute_hits[visitor], _day_hits[visitor]
    while minute and now - minute[0] > 60:
        minute.popleft()
    while day and now - day[0] > 86400:
        day.popleft()
    if len(minute) >= RATE_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail="Too many questions too quickly. Please wait a minute and try again.",
            headers={"Retry-After": "60"},
        )
    if len(day) >= RATE_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail="Daily question limit reached. Please try again tomorrow.",
            headers={"Retry-After": "3600"},
        )
    minute.append(now)
    day.append(now)
    # keep memory small: forget visitors who have gone quiet
    if len(_day_hits) > 5000:
        for key in [k for k, v in _day_hits.items() if not v or now - v[-1] > 86400]:
            _day_hits.pop(key, None)
            _minute_hits.pop(key, None)


def validate_payload(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    unknown = set(payload) - ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        raise HTTPException(
            status_code=400, detail=f"Unsupported field: {sorted(unknown)[0]}"
        )

    contents = payload.get("contents")
    if not isinstance(contents, list) or not contents:
        raise HTTPException(
            status_code=400, detail="Missing or invalid conversation contents"
        )
    if len(contents) > MAX_CONTENT_TURNS:
        raise HTTPException(status_code=400, detail="Conversation is too long")

    tools = payload.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise HTTPException(status_code=400, detail="Invalid tools")
        for tool in tools:
            if not isinstance(tool, dict) or set(tool) - ALLOWED_TOOL_KEYS:
                raise HTTPException(status_code=400, detail="Unsupported tool")


@app.get("/")
async def home():
    return {
        "message": "Welcome to Varanasya AI!",
        "status": "Backend is running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.post("/chat")
async def chat(request: Request):
    require_gateway(request)
    check_rate_limit(visitor_id(request))

    # Receive the existing Gemini request from your frontend
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON request"
        )

    validate_payload(payload)

    # Forward the request to Gemini using the private server-side API key
    gemini_url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent"

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                gemini_url,
                # Key goes in a header (not the URL) so it can never show up in logs.
                headers={"x-goog-api-key": GEMINI_API_KEY},
                json=payload
            )

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "error", {}
                ).get("message", "Gemini API request failed")
            except Exception:
                error_message = "Gemini API request failed"

            raise HTTPException(
                status_code=response.status_code,
                detail=error_message
            )

        # Return Gemini's original response structure.
        # Your frontend already understands this format.
        return response.json()

    except HTTPException:
        raise

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Gemini request timed out. Please try again."
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Gemini."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected backend error occurred."
        )
