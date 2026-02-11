"""
System status and configuration endpoint
Provides helpful information about Surge - PPTX configuration
"""

from fastapi import APIRouter, HTTPException
from utils.config_validator import SurgeConfig

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/status")
async def get_system_status():
    """
    Get current system status and configuration
    
    Useful for debugging configuration issues
    """
    is_valid, errors, warnings = SurgeConfig.validate_setup()
    
    # Collect API key statuses
    api_keys_status = {}
    
    # Required keys
    for key, info in SurgeConfig.REQUIRED_KEYS.items():
        status, preview = SurgeConfig.get_status(key)
        api_keys_status[key] = {
            "display_name": info["display"],
            "status": status.value,
            "configured": status.name == "CONFIGURED",
        }
    
    # Optional keys
    for key, info in SurgeConfig.OPTIONAL_KEYS.items():
        status, preview = SurgeConfig.get_status(key)
        api_keys_status[key] = {
            "display_name": info["display"],
            "status": status.value,
            "configured": status.name == "CONFIGURED",
            "optional": True,
        }
    
    return {
        "service": "Surge - PPTX",
        "status": "operational" if is_valid else "configuration_incomplete",
        "api_keys": api_keys_status,
        "errors": errors,
        "warnings": warnings,
        "rate_limiting": {
            "enabled": True,
            "info": "Enable/disable with RATE_LIMIT_ENABLED environment variable",
        },
    }


@router.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Surge - PPTX API",
    }


@router.get("/config-help")
async def get_config_help():
    """
    Get help for setting up API keys
    """
    return {
        "service": "Surge - PPTX",
        "configuration": {
            "required_keys": [
                {
                    "key": key,
                    "display": info["display"],
                    "purpose": info["purpose"],
                    "how_to_get": info["how_to"],
                }
                for key, info in SurgeConfig.REQUIRED_KEYS.items()
            ],
            "optional_keys": [
                {
                    "key": key,
                    "display": info["display"],
                    "purpose": info["purpose"],
                    "how_to_get": info["how_to"],
                    "note": info.get("note", ""),
                }
                for key, info in SurgeConfig.OPTIONAL_KEYS.items()
            ],
        },
        "setup_instructions": {
            "1_set_keys_in_env": "Update .env file with your API keys",
            "2_restart_server": "Restart the FastAPI server",
            "3_check_status": "Visit /api/v1/system/status to verify configuration",
        },
        "environment_variables": {
            "LLM_PROVIDER": "google (or 'openai', 'anthropic', 'ollama')",
            "GOOGLE_API_KEY": "Required for Google Gemini LLM",
            "OPENAI_API_KEY": "Optional - only if using OpenAI",
            "RATE_LIMIT_ENABLED": "true/false - Enable rate limiting",
            "RATE_LIMIT_CALLS": "Max calls per window (default: 100)",
            "RATE_LIMIT_WINDOW": "Time window in seconds (default: 3600)",
        },
    }
