from contextlib import asynccontextmanager
import os
import asyncio

from fastapi import FastAPI

from services.database import create_db_and_tables
from utils.get_env import get_app_data_directory_env
from utils.config_validator import setup_config_logging
from utils.model_availability import (
    check_llm_and_image_provider_api_or_model_availability,
)


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Initializes the application data directory and checks LLM model availability.

    """
    # Validate and report configuration on startup
    setup_config_logging()
    strict_startup_checks = (
        os.getenv("STRICT_STARTUP_CHECKS", "false").strip().lower() == "true"
    )

    app_data_dir = get_app_data_directory_env() or "/tmp/surge-pptx"
    os.makedirs(app_data_dir, exist_ok=True)

    # Keep startup resilient on Cloud Run: DB init can hang when networking/env is misconfigured.
    # Set STRICT_STARTUP_CHECKS=true to fail fast instead.
    db_startup_timeout_seconds = float(os.getenv("DB_STARTUP_TIMEOUT_SECONDS", "20"))
    try:
        await asyncio.wait_for(create_db_and_tables(), timeout=db_startup_timeout_seconds)
    except Exception as e:
        if strict_startup_checks:
            raise
        print(f"Startup DB warning: {e}")

    try:
        await check_llm_and_image_provider_api_or_model_availability()
    except Exception as e:
        # Do not block server startup by default on provider/config validation.
        # Enable strict behavior explicitly when fail-fast is desired.
        if strict_startup_checks:
            raise
        print(f"Startup validation warning: {e}")
    yield
