import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, UniqueConstraint
from sqlmodel import Field, SQLModel


class PromoCodeRedemptionModel(SQLModel, table=True):
    __tablename__ = "promo_code_redemptions"
    __table_args__ = (UniqueConstraint("promo_code", "user_id", name="uq_promo_code_user"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    promo_code: str = Field(foreign_key="promo_codes.code", index=True, nullable=False)
    user_id: uuid.UUID = Field(index=True, nullable=False)
    credits_awarded: int = Field(nullable=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
