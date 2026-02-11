import os
from utils.get_env import (
    get_allow_sqlite_fallback_env,
    get_app_data_directory_env,
    get_database_url_env,
    get_supabase_db_url_env,
)
from urllib.parse import urlsplit, urlunsplit, parse_qsl
import ssl


def get_database_url_and_connect_args() -> tuple[str, dict]:
    database_url = get_supabase_db_url_env() or get_database_url_env()
    if not database_url:
        allow_sqlite_fallback = (get_allow_sqlite_fallback_env() or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if allow_sqlite_fallback:
            database_url = "sqlite:///" + os.path.join(
                get_app_data_directory_env() or "/tmp/surge-pptx", "fastapi.db"
            )
        else:
            raise RuntimeError(
                "No database URL configured. Set SUPABASE_DB_URL for cloud deployment."
            )

    if database_url.startswith("sqlite://"):
        database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("mysql://"):
        database_url = database_url.replace("mysql://", "mysql+aiomysql://", 1)
    else:
        database_url = database_url

    try:
        split_result = urlsplit(database_url)
    except ValueError as exc:
        raise RuntimeError(
            "Database URL is malformed. If your password has special characters "
            "(@, :, /, ?, #, [, ]), URL-encode it before putting it in SUPABASE_DB_URL."
        ) from exc

    try:
        hostname = split_result.hostname
        if not hostname:
            raise RuntimeError(
                "Database URL is invalid: hostname is missing. "
                "Copy SUPABASE_DB_URL directly from Supabase Database -> Connection string."
            )
        if "<" in hostname or ">" in hostname:
            raise RuntimeError(
                "Database URL contains placeholder hostname. "
                "Replace it with the real Supabase host from the dashboard."
            )
        # Supabase Postgres requires SSL; enforce it when missing.
        if split_result.scheme.startswith("postgresql") and hostname.endswith("supabase.co"):
            query_params = dict(parse_qsl(split_result.query, keep_blank_values=True))
            if "sslmode" not in {k.lower() for k in query_params.keys()}:
                query_params["sslmode"] = "require"
                query_str = "&".join(f"{k}={v}" for k, v in query_params.items())
                database_url = urlunsplit(
                    (
                        split_result.scheme,
                        split_result.netloc,
                        split_result.path,
                        query_str,
                        split_result.fragment,
                    )
                )
    except Exception:
        raise

    connect_args = {}
    if "sqlite" in database_url:
        connect_args["check_same_thread"] = False

    try:
        split_result = urlsplit(database_url)
        if split_result.query:
            query_params = parse_qsl(split_result.query, keep_blank_values=True)
            driver_scheme = split_result.scheme
            for k, v in query_params:
                key_lower = k.lower()
                if key_lower == "sslmode" and "postgresql+asyncpg" in driver_scheme:
                    if v.lower() != "disable" and "sqlite" not in database_url:
                        connect_args["ssl"] = ssl.create_default_context()

            database_url = urlunsplit(
                (
                    split_result.scheme,
                    split_result.netloc,
                    split_result.path,
                    "",
                    split_result.fragment,
                )
            )
    except Exception:
        pass

    return database_url, connect_args
