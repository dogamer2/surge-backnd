import asyncio
import re
import smtplib
import uuid
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any
from urllib.parse import quote

import aiohttp
from fastapi import HTTPException, Request
from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, select

from models.sql.credit import CreditModel
from models.sql.crypto_payment_intent import CryptoPaymentIntentModel
from models.sql.promo_code import PromoCodeModel
from models.sql.promo_code_redemption import PromoCodeRedemptionModel
from models.sql.user_suggestion import UserSuggestionModel
from utils.get_env import (
    get_credit_cost_chat_env,
    get_credit_cost_essay_env,
    get_credit_cost_image_env,
    get_credit_cost_presentation_generate_env,
    get_credit_cost_presentation_update_env,
    get_credit_cost_slide_edit_env,
    get_credit_daily_claim_amount_env,
    get_credit_daily_claim_cooldown_hours_env,
    get_credit_start_balance_env,
    get_feedback_email_to_env,
    get_smtp_from_env,
    get_smtp_host_env,
    get_smtp_password_env,
    get_smtp_port_env,
    get_smtp_user_env,
)


def _env_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def _env_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


DEFAULT_START_CREDITS = _env_int(get_credit_start_balance_env(), 100)
DAILY_CLAIM_AMOUNT = _env_int(get_credit_daily_claim_amount_env(), 100)
DAILY_CLAIM_COOLDOWN_HOURS = max(1, _env_int(get_credit_daily_claim_cooldown_hours_env(), 24))
SUGGESTION_COOLDOWN_HOURS = 3
PAYMENT_WATCH_MINUTES = 5

CREDIT_COSTS = {
    "chat": _env_int(get_credit_cost_chat_env(), 1),
    "image": _env_int(get_credit_cost_image_env(), 5),
    "presentation_generate": _env_int(get_credit_cost_presentation_generate_env(), 15),
    "presentation_update": _env_int(get_credit_cost_presentation_update_env(), 3),
    "slide_edit": _env_int(get_credit_cost_slide_edit_env(), 2),
    "essay": _env_int(get_credit_cost_essay_env(), 8),
}

CREDIT_PACKAGES_CAD = {
    100: 0.99,
    500: 3.99,
    1000: 6.99,
    2000: 11.99,
}

COIN_COINGECKO_IDS = {
    "bitcoin": "bitcoin",
    "solana": "solana",
    "tron": "tron",
    "usdc": "usd-coin",
    "usdt": "tether",
    "ethereum": "ethereum",
    "litecoin": "litecoin",
}

COIN_SYMBOLS = {
    "bitcoin": "BTC",
    "solana": "SOL",
    "tron": "TRX",
    "usdc": "USDC",
    "usdt": "USDT",
    "ethereum": "ETH",
    "litecoin": "LTC",
}

_CREDIT_TABLES_ENSURED = False
_PROMO_EXPIRES_RE = re.compile(r"^\s*(\d+)\s*d?\s*$", re.IGNORECASE)


def get_request_user_id(request: Request) -> str:
    raw_user_id = request.headers.get("x-user-id") or request.headers.get("x-session-id") or "anonymous"
    raw_user_id = raw_user_id.strip() or "anonymous"
    try:
        return str(uuid.UUID(raw_user_id))
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_user_id))


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _daily_claim_status(last_daily_claim_at: datetime | None) -> dict:
    now_utc = datetime.now(timezone.utc)
    if last_daily_claim_at and last_daily_claim_at.tzinfo is None:
        last_daily_claim_at = last_daily_claim_at.replace(tzinfo=timezone.utc)
    if not last_daily_claim_at:
        return {
            "can_claim_daily": True,
            "next_daily_claim_at": None,
            "daily_claim_amount": DAILY_CLAIM_AMOUNT,
            "seconds_until_next_claim": 0,
        }

    next_claim_at = last_daily_claim_at + timedelta(hours=DAILY_CLAIM_COOLDOWN_HOURS)
    seconds_until_next_claim = max(0, int((next_claim_at - now_utc).total_seconds()))
    return {
        "can_claim_daily": now_utc >= next_claim_at,
        "next_daily_claim_at": _iso_utc(next_claim_at),
        "daily_claim_amount": DAILY_CLAIM_AMOUNT,
        "seconds_until_next_claim": seconds_until_next_claim,
    }


def _parse_expires_days(expires: str | None) -> int | None:
    if expires is None:
        return None
    raw = expires.strip()
    if not raw:
        return None
    match = _PROMO_EXPIRES_RE.match(raw)
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Promo code expires format is invalid. Use values like 1d or 30d.",
        )
    days = int(match.group(1))
    if days <= 0:
        raise HTTPException(status_code=400, detail="Promo code expires days must be greater than 0.")
    return days


def _normalize_coin(raw_coin: str) -> str:
    coin = (raw_coin or "").strip().lower()
    if coin == "etherium":
        coin = "ethereum"
    if coin not in COIN_COINGECKO_IDS:
        raise HTTPException(status_code=400, detail="Unsupported coin selected.")
    return coin


HARDCODED_WALLET_ADDRESSES = {
    "bitcoin": "12dQtCG2WEekteb7Y5s7UkC6oUi2wLEjT4",
    "solana": "7er73Q4kaGJZfpBJBUs5j1gqfk6URcg5rBp3ZB8FXefX",
    "tron": "TNPC5szrQ1D7YZ485CU2SL6pmGUkyJarez",
    # Per request, USDC/USDT payments are received on Solana.
    "usdc": "7er73Q4kaGJZfpBJBUs5j1gqfk6URcg5rBp3ZB8FXefX",
    "usdt": "7er73Q4kaGJZfpBJBUs5j1gqfk6URcg5rBp3ZB8FXefX",
    "ethereum": "0x2A79C7Ac15D342E6f03DCf7bCd74578f416B0Cd1",
    "litecoin": "LMXGpDDCuPFppoBarrpdPHQmqrFFciXUCs",
}

SOLANA_TOKEN_MINTS = {
    "usdc": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "usdt": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
}


def _wallet_address_for_coin(coin: str) -> str:
    address = (HARDCODED_WALLET_ADDRESSES.get(coin) or "").strip()
    if not address:
        raise HTTPException(
            status_code=500,
            detail=f"Wallet address for {coin} is empty. Set HARDCODED_WALLET_ADDRESSES in credit_service.py.",
        )
    return address


def _qrcode_payload(coin: str, address: str, amount: float) -> str:
    symbol = COIN_SYMBOLS.get(coin, coin.upper()).lower()
    if coin in {"usdt", "usdc"}:
        return f"solana:{address}?amount={amount:.8f}"
    return f"{symbol}:{address}?amount={amount:.8f}"


async def _fetch_json(
    url: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(method, url, params=params, headers=headers, json=body) as response:
            if response.status >= 400:
                text_payload = await response.text()
                raise HTTPException(
                    status_code=502,
                    detail=f"Upstream pricing/chain provider error ({response.status}): {text_payload[:200]}",
                )
            return await response.json()


async def _coin_price_in_cad(coin: str) -> float:
    gecko_id = COIN_COINGECKO_IDS[coin]
    data = await _fetch_json(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": gecko_id, "vs_currencies": "cad"},
    )
    cad_price = _env_float(str((data.get(gecko_id) or {}).get("cad")), 0.0)
    if cad_price <= 0:
        raise HTTPException(status_code=502, detail=f"Unable to quote {coin} in CAD right now.")
    return cad_price


def _amount_matches(observed: float, expected: float) -> bool:
    tolerance = max(0.000001, expected * 0.01)
    return abs(observed - expected) <= tolerance


async def _detect_blockcypher_payment(
    *,
    chain: str,
    address: str,
    expected_amount: float,
    created_after: datetime,
) -> tuple[str, float] | None:
    data = await _fetch_json(
        f"https://api.blockcypher.com/v1/{chain}/main/addrs/{address}/full",
        params={"limit": "50"},
    )
    txrefs = []
    txrefs.extend(data.get("txrefs") or [])
    txrefs.extend(data.get("unconfirmed_txrefs") or [])

    divisor = {
        "btc": 100_000_000,
        "ltc": 100_000_000,
        "eth": 1_000_000_000_000_000_000,
    }[chain]

    for tx in txrefs:
        observed = _env_float(str(tx.get("value")), 0.0) / divisor
        tx_hash = (tx.get("tx_hash") or "").strip()
        confirmed_at = _parse_iso(tx.get("confirmed") or tx.get("received"))
        if not tx_hash or not confirmed_at:
            continue
        if confirmed_at < created_after:
            continue
        if _amount_matches(observed, expected_amount):
            return tx_hash, observed
    return None


async def _detect_tron_payment(
    *,
    address: str,
    expected_amount: float,
    created_after: datetime,
) -> tuple[str, float] | None:
    data = await _fetch_json(
        f"https://api.trongrid.io/v1/accounts/{address}/transactions",
        params={"only_to": "true", "limit": "50", "order_by": "block_timestamp,desc"},
    )
    for tx in data.get("data") or []:
        tx_hash = (tx.get("txID") or "").strip()
        block_ms = tx.get("block_timestamp")
        if not tx_hash or block_ms is None:
            continue
        try:
            tx_time = datetime.fromtimestamp(int(block_ms) / 1000, tz=timezone.utc)
        except Exception:
            continue
        if tx_time < created_after:
            continue

        contracts = (((tx.get("raw_data") or {}).get("contract")) or [])
        if not contracts:
            continue
        amount_sun = ((((contracts[0] or {}).get("parameter") or {}).get("value") or {}).get("amount"))
        observed = _env_float(str(amount_sun), 0.0) / 1_000_000
        if _amount_matches(observed, expected_amount):
            return tx_hash, observed
    return None


async def _detect_solana_payment(
    *,
    address: str,
    expected_amount: float,
    created_after: datetime,
) -> tuple[str, float] | None:
    signatures_payload = await _fetch_json(
        "https://api.mainnet-beta.solana.com",
        method="POST",
        body={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [address, {"limit": 20}],
        },
    )
    signatures = (signatures_payload.get("result") or [])
    for entry in signatures:
        sig = entry.get("signature")
        block_time = entry.get("blockTime")
        if not sig or not block_time:
            continue
        tx_time = datetime.fromtimestamp(int(block_time), tz=timezone.utc)
        if tx_time < created_after:
            continue

        tx_payload = await _fetch_json(
            "https://api.mainnet-beta.solana.com",
            method="POST",
            body={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [sig, {"encoding": "json", "maxSupportedTransactionVersion": 0}],
            },
        )
        tx_result = tx_payload.get("result") or {}
        meta = tx_result.get("meta") or {}
        transaction = tx_result.get("transaction") or {}
        message = transaction.get("message") or {}
        account_keys = message.get("accountKeys") or []
        pre_balances = meta.get("preBalances") or []
        post_balances = meta.get("postBalances") or []
        if not account_keys or len(pre_balances) != len(post_balances):
            continue

        observed = 0.0
        for idx, key in enumerate(account_keys):
            if key == address:
                delta_lamports = int(post_balances[idx]) - int(pre_balances[idx])
                if delta_lamports > 0:
                    observed = delta_lamports / 1_000_000_000
                    break
        if observed > 0 and _amount_matches(observed, expected_amount):
            return sig, observed
    return None


def _ui_token_amount(balance_entry: dict[str, Any]) -> float:
    ui_token = (balance_entry.get("uiTokenAmount") or {})
    raw = ui_token.get("uiAmountString")
    if raw is None:
        raw = ui_token.get("uiAmount")
    return _env_float(str(raw), 0.0)


async def _detect_solana_spl_token_payment(
    *,
    coin: str,
    owner_address: str,
    expected_amount: float,
    created_after: datetime,
) -> tuple[str, float] | None:
    mint = SOLANA_TOKEN_MINTS.get(coin)
    if not mint:
        return None

    signatures_payload = await _fetch_json(
        "https://api.mainnet-beta.solana.com",
        method="POST",
        body={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [owner_address, {"limit": 20}],
        },
    )
    signatures = signatures_payload.get("result") or []

    for entry in signatures:
        sig = entry.get("signature")
        block_time = entry.get("blockTime")
        if not sig or not block_time:
            continue
        tx_time = datetime.fromtimestamp(int(block_time), tz=timezone.utc)
        if tx_time < created_after:
            continue

        tx_payload = await _fetch_json(
            "https://api.mainnet-beta.solana.com",
            method="POST",
            body={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [sig, {"encoding": "json", "maxSupportedTransactionVersion": 0}],
            },
        )
        tx_result = tx_payload.get("result") or {}
        meta = tx_result.get("meta") or {}
        pre_tokens = meta.get("preTokenBalances") or []
        post_tokens = meta.get("postTokenBalances") or []

        pre_by_idx: dict[int, float] = {}
        post_by_idx: dict[int, float] = {}

        for bal in pre_tokens:
            if (bal.get("owner") or "") != owner_address:
                continue
            if (bal.get("mint") or "") != mint:
                continue
            idx = int(bal.get("accountIndex") or -1)
            if idx >= 0:
                pre_by_idx[idx] = _ui_token_amount(bal)

        for bal in post_tokens:
            if (bal.get("owner") or "") != owner_address:
                continue
            if (bal.get("mint") or "") != mint:
                continue
            idx = int(bal.get("accountIndex") or -1)
            if idx >= 0:
                post_by_idx[idx] = _ui_token_amount(bal)

        if not post_by_idx:
            continue

        observed_delta = 0.0
        for idx, post_amount in post_by_idx.items():
            observed_delta += max(0.0, post_amount - pre_by_idx.get(idx, 0.0))

        if observed_delta > 0 and _amount_matches(observed_delta, expected_amount):
            return sig, observed_delta

    return None


async def _detect_matching_payment(
    *,
    coin: str,
    address: str,
    expected_amount: float,
    created_after: datetime,
) -> tuple[str, float] | None:
    if coin == "bitcoin":
        return await _detect_blockcypher_payment(
            chain="btc",
            address=address,
            expected_amount=expected_amount,
            created_after=created_after,
        )
    if coin == "litecoin":
        return await _detect_blockcypher_payment(
            chain="ltc",
            address=address,
            expected_amount=expected_amount,
            created_after=created_after,
        )
    if coin == "ethereum":
        return await _detect_blockcypher_payment(
            chain="eth",
            address=address,
            expected_amount=expected_amount,
            created_after=created_after,
        )
    if coin == "tron":
        return await _detect_tron_payment(
            address=address,
            expected_amount=expected_amount,
            created_after=created_after,
        )
    if coin == "solana":
        return await _detect_solana_payment(
            address=address,
            expected_amount=expected_amount,
            created_after=created_after,
        )
    if coin in {"usdc", "usdt"}:
        return await _detect_solana_spl_token_payment(
            coin=coin,
            owner_address=address,
            expected_amount=expected_amount,
            created_after=created_after,
        )
    return None


async def set_rls_user_context(sql_session: AsyncSession, user_id: str) -> None:
    bind = sql_session.get_bind()
    dialect_name = bind.dialect.name if bind is not None else ""
    if dialect_name != "postgresql":
        return

    await sql_session.execute(
        text("select set_config('app.user_id', :user_id, true)"),
        {"user_id": user_id},
    )
    await sql_session.execute(
        text("select set_config('request.jwt.claim.sub', :user_id, true)"),
        {"user_id": user_id},
    )
    await sql_session.execute(text("select set_config('request.jwt.claim.role', 'authenticated', true)"))


async def _drop_legacy_promo_tables_if_needed(sql_session: AsyncSession) -> None:
    bind = sql_session.get_bind()
    dialect_name = bind.dialect.name if bind is not None else ""
    if dialect_name != "postgresql":
        return

    legacy_column_query = text(
        """
        select exists (
          select 1
          from information_schema.columns
          where table_schema = 'public'
            and table_name = 'promo_codes'
            and column_name in ('is_active', 'max_redemptions', 'redeemed_count', 'expires_at')
        )
        """
    )
    legacy_result = await sql_session.execute(legacy_column_query)
    is_legacy = bool(legacy_result.scalar())
    if not is_legacy:
        return

    await sql_session.execute(text("drop table if exists public.promo_code_redemptions"))
    await sql_session.execute(text("drop table if exists public.promo_codes"))
    await sql_session.commit()


async def ensure_credit_tables(sql_session: AsyncSession) -> None:
    global _CREDIT_TABLES_ENSURED
    if _CREDIT_TABLES_ENSURED:
        return
    bind = sql_session.get_bind()
    if bind is None:
        return
    await _drop_legacy_promo_tables_if_needed(sql_session)
    await sql_session.run_sync(
        lambda sync_session: SQLModel.metadata.create_all(
            sync_session.connection(),
            tables=[
                CreditModel.__table__,
                PromoCodeModel.__table__,
                PromoCodeRedemptionModel.__table__,
                CryptoPaymentIntentModel.__table__,
                UserSuggestionModel.__table__,
            ],
        )
    )
    _CREDIT_TABLES_ENSURED = True


async def get_or_create_credits(sql_session: AsyncSession, user_id: str) -> CreditModel:
    await ensure_credit_tables(sql_session)
    await set_rls_user_context(sql_session, user_id)
    user_uuid = uuid.UUID(user_id)
    query = select(CreditModel).where(CreditModel.user_id == user_uuid)
    result = await sql_session.execute(query)
    row = result.scalars().first()
    if row:
        return row

    now_utc = datetime.now(timezone.utc)
    entry = CreditModel(
        user_id=user_uuid,
        balance=DEFAULT_START_CREDITS,
        created_at=now_utc,
        updated_at=now_utc,
    )
    sql_session.add(entry)
    await sql_session.commit()
    await sql_session.refresh(entry)
    return entry


async def get_credit_balance(sql_session: AsyncSession, user_id: str) -> int:
    entry = await get_or_create_credits(sql_session, user_id)
    return int(entry.balance)


async def get_daily_claim_status(sql_session: AsyncSession, user_id: str) -> dict:
    entry = await get_or_create_credits(sql_session, user_id)
    return _daily_claim_status(entry.last_daily_claim_at)


async def claim_daily_credits(sql_session: AsyncSession, user_id: str) -> dict:
    entry = await get_or_create_credits(sql_session, user_id)
    status = _daily_claim_status(entry.last_daily_claim_at)
    if not status["can_claim_daily"]:
        return {
            "claimed": False,
            "balance": int(entry.balance),
            **status,
        }

    now_utc = datetime.now(timezone.utc)
    entry.balance = int(entry.balance) + DAILY_CLAIM_AMOUNT
    entry.last_daily_claim_at = now_utc
    entry.updated_at = now_utc
    sql_session.add(entry)
    await sql_session.commit()
    await sql_session.refresh(entry)

    return {
        "claimed": True,
        "balance": int(entry.balance),
        **_daily_claim_status(entry.last_daily_claim_at),
    }


async def consume_credits(
    sql_session: AsyncSession, user_id: str, action: str, amount: int | None = None
) -> int:
    cost = amount if amount is not None else CREDIT_COSTS.get(action, 0)
    if cost <= 0:
        return await get_credit_balance(sql_session, user_id)

    entry = await get_or_create_credits(sql_session, user_id)
    current_balance = int(entry.balance)
    if current_balance < cost:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Need {cost}, available {current_balance}.",
        )

    entry.balance = current_balance - cost
    entry.updated_at = datetime.now(timezone.utc)
    sql_session.add(entry)
    await sql_session.commit()
    await sql_session.refresh(entry)
    return int(entry.balance)


async def redeem_promo_code(sql_session: AsyncSession, user_id: str, promo_code: str) -> dict:
    normalized_code = (promo_code or "").strip()
    if not normalized_code:
        raise HTTPException(status_code=400, detail="Promo code is required.")

    await ensure_credit_tables(sql_session)
    await set_rls_user_context(sql_session, user_id)
    user_uuid = uuid.UUID(user_id)

    promo_query = select(PromoCodeModel).where(
        func.lower(PromoCodeModel.code) == normalized_code.lower()
    )
    promo_result = await sql_session.execute(promo_query)
    promo = promo_result.scalars().first()
    if not promo:
        raise HTTPException(status_code=404, detail="Promo code not found.")

    expires_days = _parse_expires_days(promo.expires)
    if expires_days is not None:
        created_at = promo.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) >= created_at + timedelta(days=expires_days):
            raise HTTPException(status_code=400, detail="Promo code has expired.")

    if promo.times is not None:
        usage_count_query = select(func.count(PromoCodeRedemptionModel.id)).where(
            PromoCodeRedemptionModel.promo_code == promo.code
        )
        usage_count_result = await sql_session.execute(usage_count_query)
        usage_count = int(usage_count_result.scalar() or 0)
        if usage_count >= int(promo.times):
            raise HTTPException(status_code=400, detail="Promo code has reached its redemption limit.")

    existing_query = select(PromoCodeRedemptionModel).where(
        PromoCodeRedemptionModel.promo_code == promo.code,
        PromoCodeRedemptionModel.user_id == user_uuid,
    )
    existing_result = await sql_session.execute(existing_query)
    existing_redemption = existing_result.scalars().first()
    if existing_redemption:
        raise HTTPException(status_code=409, detail="Promo code already redeemed.")

    promo_credit_amount = int(promo.credit_amount)
    if promo_credit_amount <= 0:
        raise HTTPException(status_code=400, detail="Promo code credit amount must be greater than 0.")

    credits = await get_or_create_credits(sql_session, user_id)
    credits.balance = int(credits.balance) + promo_credit_amount
    credits.updated_at = datetime.now(timezone.utc)
    sql_session.add(credits)

    redemption = PromoCodeRedemptionModel(
        promo_code=promo.code,
        user_id=user_uuid,
        credits_awarded=promo_credit_amount,
    )
    sql_session.add(redemption)

    await sql_session.commit()
    await sql_session.refresh(credits)

    return {
        "code": promo.code,
        "awarded_credits": promo_credit_amount,
        "balance": int(credits.balance),
    }


async def create_crypto_payment_intent(
    sql_session: AsyncSession,
    *,
    user_id: str,
    coin: str,
    credits_to_buy: int,
) -> dict:
    await ensure_credit_tables(sql_session)
    await set_rls_user_context(sql_session, user_id)

    normalized_coin = _normalize_coin(coin)
    cad_amount = CREDIT_PACKAGES_CAD.get(int(credits_to_buy))
    if cad_amount is None:
        raise HTTPException(status_code=400, detail="Unsupported credit package.")

    wallet_address = _wallet_address_for_coin(normalized_coin)
    cad_price = await _coin_price_in_cad(normalized_coin)
    coin_amount = round(cad_amount / cad_price, 8)
    if coin_amount <= 0:
        raise HTTPException(status_code=502, detail="Unable to compute coin amount for selected package.")

    now_utc = datetime.now(timezone.utc)
    expires_at = now_utc + timedelta(minutes=PAYMENT_WATCH_MINUTES)
    intent = CryptoPaymentIntentModel(
        user_id=uuid.UUID(user_id),
        coin=normalized_coin,
        credits_to_grant=int(credits_to_buy),
        cad_amount=float(cad_amount),
        coin_amount=float(coin_amount),
        wallet_address=wallet_address,
        expires_at=expires_at,
        status="pending",
        created_at=now_utc,
        updated_at=now_utc,
    )
    sql_session.add(intent)
    await sql_session.commit()
    await sql_session.refresh(intent)

    qr_payload = _qrcode_payload(normalized_coin, wallet_address, coin_amount)
    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=320x320&data={quote(qr_payload, safe='')}"

    return {
        "intent_id": str(intent.id),
        "coin": normalized_coin,
        "coin_symbol": COIN_SYMBOLS[normalized_coin],
        "credits_to_grant": int(intent.credits_to_grant),
        "cad_amount": float(intent.cad_amount),
        "coin_amount": float(intent.coin_amount),
        "wallet_address": intent.wallet_address,
        "status": intent.status,
        "expires_at": _iso_utc(intent.expires_at),
        "qr_payload": qr_payload,
        "qr_url": qr_url,
    }


async def _apply_detected_payment(
    sql_session: AsyncSession,
    *,
    user_id: str,
    intent: CryptoPaymentIntentModel,
    tx_hash: str,
) -> None:
    already_used_query = select(CryptoPaymentIntentModel).where(
        CryptoPaymentIntentModel.detected_tx_hash == tx_hash,
        CryptoPaymentIntentModel.id != intent.id,
    )
    already_used_result = await sql_session.execute(already_used_query)
    if already_used_result.scalars().first():
        raise HTTPException(status_code=409, detail="Transaction already used for another payment intent.")

    credits = await get_or_create_credits(sql_session, user_id)
    credits.balance = int(credits.balance) + int(intent.credits_to_grant)
    credits.updated_at = datetime.now(timezone.utc)

    intent.status = "paid"
    intent.detected_tx_hash = tx_hash
    intent.provider_message = "Transaction detected and credits granted."
    intent.updated_at = datetime.now(timezone.utc)

    sql_session.add(credits)
    sql_session.add(intent)
    await sql_session.commit()
    try:
        await _send_deposit_notification(
            user_id=user_id,
            coin=intent.coin,
            coin_amount=float(intent.coin_amount),
            cad_amount=float(intent.cad_amount),
            credits_granted=int(intent.credits_to_grant),
            tx_hash=tx_hash,
        )
    except Exception:
        # Do not fail credit grants because of email delivery issues.
        pass


async def get_crypto_payment_intent_status(
    sql_session: AsyncSession,
    *,
    user_id: str,
    intent_id: str,
) -> dict:
    await ensure_credit_tables(sql_session)
    await set_rls_user_context(sql_session, user_id)

    try:
        parsed_intent_id = uuid.UUID(intent_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid intent id.")

    intent_query = select(CryptoPaymentIntentModel).where(
        CryptoPaymentIntentModel.id == parsed_intent_id,
        CryptoPaymentIntentModel.user_id == uuid.UUID(user_id),
    )
    intent_result = await sql_session.execute(intent_query)
    intent = intent_result.scalars().first()
    if not intent:
        raise HTTPException(status_code=404, detail="Payment intent not found.")

    now_utc = datetime.now(timezone.utc)

    if intent.status == "pending":
        if now_utc >= intent.expires_at:
            intent.status = "expired"
            intent.provider_message = "Payment watch window expired."
            intent.updated_at = now_utc
            sql_session.add(intent)
            await sql_session.commit()
        else:
            try:
                detection = await _detect_matching_payment(
                    coin=intent.coin,
                    address=intent.wallet_address,
                    expected_amount=float(intent.coin_amount),
                    created_after=intent.created_at,
                )
            except HTTPException as exc:
                intent.provider_message = exc.detail
                intent.updated_at = now_utc
                sql_session.add(intent)
                await sql_session.commit()
                detection = None

            if detection:
                tx_hash, observed_amount = detection
                await _apply_detected_payment(
                    sql_session,
                    user_id=user_id,
                    intent=intent,
                    tx_hash=tx_hash,
                )
                intent.coin_amount = float(observed_amount)

    refreshed_query = select(CryptoPaymentIntentModel).where(CryptoPaymentIntentModel.id == parsed_intent_id)
    refreshed_result = await sql_session.execute(refreshed_query)
    refreshed = refreshed_result.scalars().first()
    if not refreshed:
        raise HTTPException(status_code=404, detail="Payment intent disappeared.")

    balance = await get_credit_balance(sql_session, user_id)
    return {
        "intent_id": str(refreshed.id),
        "status": refreshed.status,
        "coin": refreshed.coin,
        "coin_symbol": COIN_SYMBOLS.get(refreshed.coin, refreshed.coin.upper()),
        "credits_to_grant": int(refreshed.credits_to_grant),
        "cad_amount": float(refreshed.cad_amount),
        "coin_amount": float(refreshed.coin_amount),
        "wallet_address": refreshed.wallet_address,
        "detected_tx_hash": refreshed.detected_tx_hash,
        "provider_message": refreshed.provider_message,
        "expires_at": _iso_utc(refreshed.expires_at),
        "balance": balance,
    }


def _send_email_sync(subject: str, body: str, to_email: str) -> None:
    smtp_host = (get_smtp_host_env() or "").strip()
    smtp_port = _env_int(get_smtp_port_env(), 587)
    smtp_user = (get_smtp_user_env() or "").strip()
    smtp_password = (get_smtp_password_env() or "").strip()
    smtp_from = (get_smtp_from_env() or smtp_user).strip()

    if not smtp_host or not smtp_user or not smtp_password or not smtp_from:
        raise HTTPException(
            status_code=500,
            detail="SMTP credentials are not configured. Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM.",
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


async def _send_deposit_notification(
    *,
    user_id: str,
    coin: str,
    coin_amount: float,
    cad_amount: float,
    credits_granted: int,
    tx_hash: str,
) -> None:
    to_email = (get_feedback_email_to_env() or "jhwmelhuish@gmail.com").strip()
    subject = "Surge Crypto Deposit Confirmed"
    body = (
        "A user deposit was confirmed.\n\n"
        f"User: {user_id}\n"
        f"Coin: {coin}\n"
        f"Coin amount: {coin_amount}\n"
        f"CAD amount: {cad_amount}\n"
        f"Credits granted: {credits_granted}\n"
        f"Transaction hash: {tx_hash}\n"
        f"Time (UTC): {_iso_utc(datetime.now(timezone.utc))}\n"
    )
    await asyncio.to_thread(_send_email_sync, subject, body, to_email)


async def submit_user_suggestion(sql_session: AsyncSession, *, user_id: str, message: str) -> dict:
    await ensure_credit_tables(sql_session)
    await set_rls_user_context(sql_session, user_id)

    normalized_message = (message or "").strip()
    if len(normalized_message) < 5:
        raise HTTPException(status_code=400, detail="Message is too short.")
    if len(normalized_message) > 5000:
        raise HTTPException(status_code=400, detail="Message is too long.")

    user_uuid = uuid.UUID(user_id)
    latest_query = (
        select(UserSuggestionModel)
        .where(UserSuggestionModel.user_id == user_uuid)
        .order_by(UserSuggestionModel.created_at.desc())
        .limit(1)
    )
    latest_result = await sql_session.execute(latest_query)
    latest = latest_result.scalars().first()

    now_utc = datetime.now(timezone.utc)
    if latest and latest.created_at + timedelta(hours=SUGGESTION_COOLDOWN_HOURS) > now_utc:
        next_at = latest.created_at + timedelta(hours=SUGGESTION_COOLDOWN_HOURS)
        seconds = int((next_at - now_utc).total_seconds())
        raise HTTPException(
            status_code=429,
            detail=f"Please wait before sending another message.",
            headers={"Retry-After": str(max(1, seconds))},
        )

    to_email = (get_feedback_email_to_env() or "jhwmelhuish@gmail.com").strip()
    subject = "Surge Suggestion / Contact"
    body = f"User: {user_id}\nTime (UTC): {_iso_utc(now_utc)}\n\n{normalized_message}"

    await asyncio.to_thread(_send_email_sync, subject, body, to_email)

    row = UserSuggestionModel(
        user_id=user_uuid,
        email_to=to_email,
        message=normalized_message,
        created_at=now_utc,
    )
    sql_session.add(row)
    await sql_session.commit()

    return {
        "sent": True,
        "cooldown_hours": SUGGESTION_COOLDOWN_HOURS,
    }
