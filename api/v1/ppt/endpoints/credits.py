from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.database import get_async_session
from services.credit_service import (
    claim_daily_credits,
    create_crypto_payment_intent,
    get_credit_balance,
    get_crypto_payment_intent_status,
    get_daily_claim_status,
    get_request_user_id,
    redeem_promo_code,
    submit_user_suggestion,
)


CREDITS_ROUTER = APIRouter(prefix="/credits", tags=["Credits"])


class RedeemPromoCodeRequest(BaseModel):
    code: str


class CreateCryptoIntentRequest(BaseModel):
    coin: str
    credits: int


class SuggestionRequest(BaseModel):
    message: str


@CREDITS_ROUTER.get("/balance")
async def get_balance(
    request: Request,
    sql_session: AsyncSession = Depends(get_async_session),
):
    user_id = get_request_user_id(request)
    balance = await get_credit_balance(sql_session, user_id)
    daily_claim = await get_daily_claim_status(sql_session, user_id)
    return {"user_id": user_id, "balance": balance, **daily_claim}


@CREDITS_ROUTER.post("/claim-daily")
async def claim_daily(
    request: Request,
    sql_session: AsyncSession = Depends(get_async_session),
):
    user_id = get_request_user_id(request)
    claim_result = await claim_daily_credits(sql_session, user_id)
    return {"user_id": user_id, **claim_result}


@CREDITS_ROUTER.post("/redeem-promo")
async def redeem_promo(
    payload: RedeemPromoCodeRequest,
    request: Request,
    sql_session: AsyncSession = Depends(get_async_session),
):
    user_id = get_request_user_id(request)
    redemption = await redeem_promo_code(sql_session, user_id, payload.code)
    return {"user_id": user_id, **redemption}


@CREDITS_ROUTER.post("/crypto/create-intent")
async def create_crypto_intent(
    payload: CreateCryptoIntentRequest,
    request: Request,
    sql_session: AsyncSession = Depends(get_async_session),
):
    user_id = get_request_user_id(request)
    result = await create_crypto_payment_intent(
        sql_session,
        user_id=user_id,
        coin=payload.coin,
        credits_to_buy=payload.credits,
    )
    return {"user_id": user_id, **result}


@CREDITS_ROUTER.get("/crypto/intent/{intent_id}")
async def get_crypto_intent_status(
    intent_id: str,
    request: Request,
    sql_session: AsyncSession = Depends(get_async_session),
):
    user_id = get_request_user_id(request)
    result = await get_crypto_payment_intent_status(
        sql_session,
        user_id=user_id,
        intent_id=intent_id,
    )
    return {"user_id": user_id, **result}


@CREDITS_ROUTER.post("/suggestions")
async def submit_suggestion(
    payload: SuggestionRequest,
    request: Request,
    sql_session: AsyncSession = Depends(get_async_session),
):
    user_id = get_request_user_id(request)
    result = await submit_user_suggestion(
        sql_session,
        user_id=user_id,
        message=payload.message,
    )
    return {"user_id": user_id, **result}
