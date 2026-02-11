from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


class PromoCodeModel(SQLModel, table=True):
    __tablename__ = "promo_codes"

    # User-managed fields:
    # - code: promo code text
    # - times: max number of unique users that can redeem
    # - credit_amount: credits this promo gives on redemption
    # - expires: duration string (e.g. "1d", "30d"), counted from created_at
    code: str = Field(primary_key=True, index=True, nullable=False)
    times: Optional[int] = Field(default=None)
    credit_amount: int = Field(nullable=False)
    expires: Optional[str] = Field(default=None, nullable=True)

    # System-managed timestamp used to evaluate `expires` duration.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
