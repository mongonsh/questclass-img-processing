"""
QuestClass 3D Backend
FastAPI server — receives DALL-E images, runs SAM + SAM 3D, returns asset URLs.

Start:
  source .venv/bin/activate
  uvicorn main:app --reload --port 8000
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import stripe

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

app = FastAPI(title="QuestClass 3D", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    image_base64: str   # raw base64 PNG (no data: prefix)


class ProcessResponse(BaseModel):
    sprite_url: str          # always present — SAM-segmented transparent PNG
    ply_url: str | None      # Gaussian splat PLY, None if SAM 3D not available
    success: bool


class OpenAIProxyRequest(BaseModel):
    payload: dict[str, Any]


class CheckoutSessionRequest(BaseModel):
    app_user_id: str
    email: str | None = None
    plan: str
    billing_cycle: str
    success_url: str
    cancel_url: str


class CheckoutSessionResponse(BaseModel):
    checkout_url: str
    session_id: str


class ConfirmCheckoutSessionRequest(BaseModel):
    session_id: str
    app_user_id: str


def _openai_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the backend")
    return api_key


def _env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(status_code=500, detail=f"{name} is not configured on the backend")
    return value


def _stripe_price_id(plan: str, billing_cycle: str) -> str:
    normalized_plan = plan.strip().upper()
    normalized_cycle = billing_cycle.strip().upper()
    env_name = f"STRIPE_PRICE_{normalized_plan}_{normalized_cycle}"
    value = _env_required(env_name)
    if value.startswith("price_"):
        return value

    if not value.startswith("prod_"):
        raise HTTPException(
            status_code=500,
            detail=f"{env_name} must be a Stripe price ID (price_...) or product ID (prod_...)",
        )

    try:
        product = stripe.Product.retrieve(value)
    except Exception as exc:
        logging.error("Failed to resolve Stripe product %s for %s: %s", value, env_name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not load Stripe product for {env_name}")

    default_price = getattr(product, "default_price", None)
    if isinstance(default_price, str) and default_price.startswith("price_"):
        return default_price
    if default_price and getattr(default_price, "id", "").startswith("price_"):
        return default_price.id

    raise HTTPException(
        status_code=500,
        detail=f"{env_name} points to a Stripe product without a default recurring price",
    )


def _revenuecat_public_key() -> str:
    return _env_required("REVENUECAT_STRIPE_PUBLIC_API_KEY")


def _revenuecat_public_key_optional() -> str | None:
    value = os.getenv("REVENUECAT_STRIPE_PUBLIC_API_KEY", "").strip()
    return value or None


def _configure_stripe() -> None:
    stripe.api_key = _env_required("STRIPE_SECRET_KEY")


def _firebase_db():
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        key_path = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
        if os.path.exists(key_path):
            cred = credentials.Certificate(key_path)
        else:
            cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(
            cred,
            {"storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "").strip()},
        )

    return firestore.client()


def _mark_user_tier(app_user_id: str, tier: str = "pro") -> None:
    try:
        db = _firebase_db()
        db.collection("users").document(app_user_id).set({"tier": tier}, merge=True)
    except Exception as exc:
        logging.warning("Failed to update user tier for %s: %s", app_user_id, exc)


async def _sync_stripe_purchase_to_revenuecat(app_user_id: str, fetch_token: str) -> dict[str, Any]:
    revenuecat_key = _revenuecat_public_key_optional()
    _mark_user_tier(app_user_id, "pro")

    if not revenuecat_key:
        logging.warning(
            "Skipping RevenueCat sync for %s because REVENUECAT_STRIPE_PUBLIC_API_KEY is not configured",
            app_user_id,
        )
        return {"skipped": True, "reason": "REVENUECAT_STRIPE_PUBLIC_API_KEY is not configured"}

    headers = {
        "Authorization": f"Bearer {revenuecat_key}",
        "Content-Type": "application/json",
        "X-Platform": "stripe",
    }
    payload = {"app_user_id": app_user_id, "fetch_token": fetch_token}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post("https://api.revenuecat.com/v1/receipts", headers=headers, json=payload)

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = {"message": response.text}
        logging.warning("RevenueCat sync failed for %s: %s", app_user_id, detail)
        return {"synced": False, "error": detail}

    return {"synced": True, "revenuecat": response.json()}


def _stripe_subscription_token_from_session(session: Any) -> str:
    subscription = getattr(session, "subscription", None)
    if isinstance(subscription, str) and subscription:
        return subscription
    if subscription and getattr(subscription, "id", None):
        return subscription.id
    if getattr(session, "id", None):
        return session.id
    raise HTTPException(status_code=400, detail="Stripe session does not include a usable subscription token")


async def _openai_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {_openai_key()}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"https://api.openai.com/v1/{path}", headers=headers, json=payload)

    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = {"message": response.text}
        raise HTTPException(status_code=response.status_code, detail=detail)

    return response.json()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process-object", response_model=ProcessResponse)
async def process_object(req: ProcessRequest):
    """
    Full pipeline:
      DALL-E base64 PNG → SAM segment → SAM 3D reconstruct → Firebase upload
    Returns permanent Firebase Storage URLs.
    """
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    from pipeline import process_object_image

    try:
        result = await process_object_image(req.image_base64)
        return ProcessResponse(**result)
    except Exception as e:
        logging.error(f"process_object error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment-only", response_model=ProcessResponse)
async def segment_only(req: ProcessRequest):
    """
    SAM segmentation only — no SAM 3D.
    Useful for testing / when GPU unavailable.
    """
    from pipeline import process_object_image
    import pipeline

    # Patch: temporarily disable SAM 3D
    original = pipeline._sam3d_inference
    pipeline._sam3d_inference = "disabled"   # truthy → skip load attempt

    try:
        result = await process_object_image(req.image_base64)
        result["ply_url"] = None
        return ProcessResponse(**result)
    finally:
        pipeline._sam3d_inference = original


@app.post("/openai/chat/completions")
async def openai_chat_completions(req: OpenAIProxyRequest):
    return await _openai_post("chat/completions", req.payload)


@app.post("/openai/images/generations")
async def openai_image_generations(req: OpenAIProxyRequest):
    return await _openai_post("images/generations", req.payload)


@app.post("/billing/create-checkout-session", response_model=CheckoutSessionResponse)
async def create_checkout_session(req: CheckoutSessionRequest):
    if not req.app_user_id:
        raise HTTPException(status_code=400, detail="app_user_id is required")

    _configure_stripe()
    price_id = _stripe_price_id(req.plan, req.billing_cycle)

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=req.success_url,
            cancel_url=req.cancel_url,
            client_reference_id=req.app_user_id,
            customer_email=req.email or None,
            allow_promotion_codes=True,
            metadata={
                "app_user_id": req.app_user_id,
                "plan": req.plan,
                "billing_cycle": req.billing_cycle,
            },
            subscription_data={
                "metadata": {
                    "app_user_id": req.app_user_id,
                    "plan": req.plan,
                    "billing_cycle": req.billing_cycle,
                }
            },
        )
    except Exception as exc:
        logging.error("Stripe checkout session creation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if not session.url:
        raise HTTPException(status_code=500, detail="Stripe Checkout did not return a URL")

    return CheckoutSessionResponse(checkout_url=session.url, session_id=session.id)


@app.post("/billing/confirm-checkout-session")
async def confirm_checkout_session(req: ConfirmCheckoutSessionRequest):
    _configure_stripe()

    try:
        session = stripe.checkout.Session.retrieve(req.session_id, expand=["subscription"])
    except Exception as exc:
        logging.error("Stripe session retrieve failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if getattr(session, "mode", None) != "subscription":
        raise HTTPException(status_code=400, detail="Stripe session is not a subscription checkout")

    if getattr(session, "payment_status", None) not in {"paid", "no_payment_required"}:
        raise HTTPException(status_code=400, detail="Stripe Checkout session is not paid yet")

    app_user_id = req.app_user_id or getattr(session, "client_reference_id", None)
    if not app_user_id:
        raise HTTPException(status_code=400, detail="No app_user_id found for Stripe session")

    receipt = await _sync_stripe_purchase_to_revenuecat(
        app_user_id=app_user_id,
        fetch_token=_stripe_subscription_token_from_session(session),
    )
    return {"success": True, "receipt_synced": True, "revenuecat": receipt, "tier_updated": True}


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    webhook_secret = _env_required("STRIPE_WEBHOOK_SECRET")
    _configure_stripe()

    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=signature, secret=webhook_secret)
    except Exception as exc:
        logging.warning("Invalid Stripe webhook: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid Stripe webhook: {exc}")

    event_type = event.get("type")
    data_object = event.get("data", {}).get("object", {})

    try:
        if event_type == "checkout.session.completed" and data_object.get("mode") == "subscription":
            app_user_id = data_object.get("client_reference_id") or data_object.get("metadata", {}).get("app_user_id")
            fetch_token = data_object.get("subscription") or data_object.get("id")
            if app_user_id and fetch_token:
                await _sync_stripe_purchase_to_revenuecat(app_user_id=app_user_id, fetch_token=fetch_token)
        elif event_type == "customer.subscription.created":
            app_user_id = data_object.get("metadata", {}).get("app_user_id")
            fetch_token = data_object.get("id")
            if app_user_id and fetch_token:
                await _sync_stripe_purchase_to_revenuecat(app_user_id=app_user_id, fetch_token=fetch_token)
    except Exception as exc:
        logging.error("Stripe webhook sync failed: %s", exc, exc_info=True)
        return {"received": True, "sync_error": str(exc)}

    return {"received": True}
