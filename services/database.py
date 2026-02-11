from collections.abc import AsyncGenerator
import os
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlmodel import SQLModel

from models.sql.async_presentation_generation_status import (
    AsyncPresentationGenerationTaskModel,
)
from models.sql.image_asset import ImageAsset
from models.sql.credit import CreditModel
from models.sql.crypto_payment_intent import CryptoPaymentIntentModel
from models.sql.key_value import KeyValueSqlModel
from models.sql.ollama_pull_status import OllamaPullStatus
from models.sql.presentation import PresentationModel
from models.sql.slide import SlideModel
from models.sql.presentation_layout_code import PresentationLayoutCodeModel
from models.sql.promo_code import PromoCodeModel
from models.sql.promo_code_redemption import PromoCodeRedemptionModel
from models.sql.template import TemplateModel
from models.sql.user_suggestion import UserSuggestionModel
from models.sql.webhook_subscription import WebhookSubscription
from utils.db_utils import get_database_url_and_connect_args
from utils.get_env import get_app_data_directory_env, get_supabase_db_url_env


database_url, connect_args = get_database_url_and_connect_args()

sql_engine: AsyncEngine = create_async_engine(database_url, connect_args=connect_args)
async_session_maker = async_sessionmaker(sql_engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


# Container DB (Lives inside the container)
# For local development, use app_data directory; in Docker, will use /app/container.db
app_data_path = get_app_data_directory_env() or "/tmp/surge-pptx"
if get_supabase_db_url_env():
    container_db_engine = sql_engine
    container_db_async_session_maker = async_session_maker
else:
    container_db_path = os.path.join(app_data_path, "container.db")
    container_db_url = "sqlite+aiosqlite:///" + container_db_path
    container_db_engine: AsyncEngine = create_async_engine(
        container_db_url, connect_args={"check_same_thread": False}
    )
    container_db_async_session_maker = async_sessionmaker(
        container_db_engine, expire_on_commit=False
    )


async def get_container_db_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with container_db_async_session_maker() as session:
        yield session


# Create Database and Tables
async def create_db_and_tables():
    async with sql_engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: SQLModel.metadata.create_all(
                sync_conn,
                tables=[
                    PresentationModel.__table__,
                    SlideModel.__table__,
                    CreditModel.__table__,
                    CryptoPaymentIntentModel.__table__,
                    KeyValueSqlModel.__table__,
                    ImageAsset.__table__,
                    PresentationLayoutCodeModel.__table__,
                    PromoCodeModel.__table__,
                    PromoCodeRedemptionModel.__table__,
                    TemplateModel.__table__,
                    UserSuggestionModel.__table__,
                    WebhookSubscription.__table__,
                    AsyncPresentationGenerationTaskModel.__table__,
                    OllamaPullStatus.__table__,
                ],
            )
        )

    if not get_supabase_db_url_env():
        async with container_db_engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: SQLModel.metadata.create_all(
                    sync_conn,
                    tables=[OllamaPullStatus.__table__],
                )
            )
