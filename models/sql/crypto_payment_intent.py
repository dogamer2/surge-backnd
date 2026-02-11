import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, UniqueConstraint
from sqlmodel import Field, SQLModel


class CryptoPaymentIntentModel(SQLModel, table=True):
    __tablename__ = "crypto_payment_intents"
    __table_args__ = (UniqueConstraint("detected_tx_hash", name="uq_crypto_detected_tx_hash"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: uuid.UUID = Field(index=True, nullable=False)
    coin: str = Field(nullable=False, index=True)
    credits_to_grant: int = Field(nullable=False)
    cad_amount: float = Field(nullable=False)
    coin_amount: float = Field(nullable=False)
    wallet_address: str = Field(nullable=False)
    status: str = Field(default="pending", nullable=False, index=True)
    detected_tx_hash: str | None = Field(default=None, nullable=True)
    provider_message: str | None = Field(default=None, nullable=True)
    expires_at: datetime = Field(sa_column=Column(DateTime(timezone=True), nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
