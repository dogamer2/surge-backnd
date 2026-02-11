from dotenv import load_dotenv
from pathlib import Path
import os
ROOT_DIR = Path(__file__).resolve().parents[2]
# Load both potential env locations:
# 1) monorepo root: /Presenton/.env
# 2) servers root: /Presenton/servers/.env (legacy/local override)
load_dotenv(ROOT_DIR.parent / ".env")
load_dotenv(ROOT_DIR / ".env", override=True)


def get_can_change_keys_env():
    return os.getenv("CAN_CHANGE_KEYS")


def get_database_url_env():
    return os.getenv("DATABASE_URL")

def get_supabase_db_url_env():
    return os.getenv("SUPABASE_DB_URL") or os.getenv("supabase_db_url")

def get_supabase_url_env():
    return os.getenv("SUPABASE_URL") or os.getenv("supabase_url")

def get_supabase_anon_key_env():
    return os.getenv("SUPABASE_ANON_KEY") or os.getenv("supabase_anon_key")

def get_supabase_service_role_key_env():
    return os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("supabase_service_role_key")

def get_allow_sqlite_fallback_env():
    return os.getenv("ALLOW_SQLITE_FALLBACK")


def get_app_data_directory_env():
    return os.getenv("APP_DATA_DIRECTORY")


def get_temp_directory_env():
    return os.getenv("TEMP_DIRECTORY")


def get_user_config_path_env():
    return os.getenv("USER_CONFIG_PATH")


def get_llm_provider_env():
    return os.getenv("LLM_PROVIDER")


def get_anthropic_api_key_env():
    return os.getenv("ANTHROPIC_API_KEY")


def get_anthropic_model_env():
    return os.getenv("ANTHROPIC_MODEL")


def get_ollama_url_env():
    return os.getenv("OLLAMA_URL")


def get_custom_llm_url_env():
    return os.getenv("CUSTOM_LLM_URL")


def get_openai_api_key_env():
    return os.getenv("OPENAI_API_KEY")


def get_openai_model_env():
    return os.getenv("OPENAI_MODEL")


def get_google_api_key_env():
    return os.getenv("GOOGLE_API_KEY")


def get_google_model_env():
    return os.getenv("GOOGLE_MODEL")


def get_custom_llm_api_key_env():
    return os.getenv("CUSTOM_LLM_API_KEY")


def get_ollama_model_env():
    return os.getenv("OLLAMA_MODEL")


def get_custom_model_env():
    return os.getenv("CUSTOM_MODEL")


def get_pexels_api_key_env():
    return os.getenv("PEXELS_API_KEY")


def get_disable_image_generation_env():
    return os.getenv("DISABLE_IMAGE_GENERATION")


def get_image_provider_env():
    return os.getenv("IMAGE_PROVIDER")


def get_pixabay_api_key_env():
    return os.getenv("PIXABAY_API_KEY")


def get_tool_calls_env():
    return os.getenv("TOOL_CALLS")


def get_disable_thinking_env():
    return os.getenv("DISABLE_THINKING")


def get_extended_reasoning_env():
    return os.getenv("EXTENDED_REASONING")


def get_web_grounding_env():
    return os.getenv("WEB_GROUNDING")


def get_comfyui_url_env():
    return os.getenv("COMFYUI_URL")


def get_comfyui_workflow_env():
    return os.getenv("COMFYUI_WORKFLOW")


# Dalle 3 Quality
def get_dall_e_3_quality_env():
    return os.getenv("DALL_E_3_QUALITY")


# Gpt Image 1.5 Quality
def get_gpt_image_1_5_quality_env():
    return os.getenv("GPT_IMAGE_1_5_QUALITY")


def get_credit_start_balance_env():
    return os.getenv("CREDIT_START_BALANCE")


def get_credit_cost_chat_env():
    return os.getenv("CREDIT_COST_CHAT")


def get_credit_cost_image_env():
    return os.getenv("CREDIT_COST_IMAGE")


def get_credit_cost_presentation_generate_env():
    return os.getenv("CREDIT_COST_PRESENTATION_GENERATE")


def get_credit_cost_presentation_update_env():
    return os.getenv("CREDIT_COST_PRESENTATION_UPDATE")


def get_credit_cost_slide_edit_env():
    return os.getenv("CREDIT_COST_SLIDE_EDIT")


def get_credit_cost_essay_env():
    return os.getenv("CREDIT_COST_ESSAY")


def get_credit_daily_claim_amount_env():
    return os.getenv("CREDIT_DAILY_CLAIM_AMOUNT")


def get_credit_daily_claim_cooldown_hours_env():
    return os.getenv("CREDIT_DAILY_CLAIM_COOLDOWN_HOURS")


def get_crypto_wallets_config_path_env():
    return os.getenv("CRYPTO_WALLETS_CONFIG_PATH")


def get_etherscan_api_key_env():
    return os.getenv("ETHERSCAN_API_KEY")


def get_feedback_email_to_env():
    return os.getenv("FEEDBACK_EMAIL_TO")


def get_smtp_host_env():
    return os.getenv("SMTP_HOST")


def get_smtp_port_env():
    return os.getenv("SMTP_PORT")


def get_smtp_user_env():
    return os.getenv("SMTP_USER")


def get_smtp_password_env():
    return os.getenv("SMTP_PASSWORD")


def get_smtp_from_env():
    return os.getenv("SMTP_FROM")
