"""
Configuration validator for Surge - PPTX
Checks required API keys and provides helpful setup guidance
"""

import os
import logging
from typing import Dict, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class APIKeyStatus(Enum):
    """Status of API key configuration"""
    CONFIGURED = "✅ Configured"
    MISSING = "❌ Missing"
    INVALID = "⚠️  Invalid"


class SurgeConfig:
    """Configuration validator for Surge - PPTX"""
    
    # Required API keys for core functionality
    REQUIRED_KEYS: Dict[str, Dict] = {
        "GOOGLE_API_KEY": {
            "display": "Google API Key",
            "purpose": "Template creation & slide generation via Gemini",
            "how_to": "https://makersuite.google.com/app/apikey",
        },
        "PEXELS_API_KEY": {
            "display": "Pexels API Key",
            "purpose": "Free stock image generation",
            "how_to": "https://www.pexels.com/api/",
        },
    }
    
    # Optional but recommended keys
    OPTIONAL_KEYS: Dict[str, Dict] = {
        "OPENAI_API_KEY": {
            "display": "OpenAI API Key",
            "purpose": "Alternative LLM for layout processing (optional if using Google)",
            "how_to": "https://platform.openai.com/api-keys",
            "note": "Only needed if you switch LLM_PROVIDER to 'openai'",
        },
    }
    
    @staticmethod
    def get_status(key: str) -> Tuple[APIKeyStatus, str]:
        """
        Get status of an API key
        
        Returns:
            (status, value_preview)
        """
        value = os.getenv(key, "").strip()
        
        if not value:
            return APIKeyStatus.MISSING, ""
        
        # Check if it looks valid (not placeholder like "your-key-here")
        if any(placeholder in value.lower() for placeholder in ["your", "replace", "example", "demo"]):
            return APIKeyStatus.INVALID, value[:10] + "..."
        
        if value and len(value) > 0:
            return APIKeyStatus.CONFIGURED, value[:10] + "..."
        
        return APIKeyStatus.MISSING, ""
    
    @staticmethod
    def validate_setup() -> Tuple[bool, List[str], List[str]]:
        """
        Validate the setup and return warnings/sections
        
        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check required keys
        for key, info in SurgeConfig.REQUIRED_KEYS.items():
            status, preview = SurgeConfig.get_status(key)
            
            if status == APIKeyStatus.MISSING:
                errors.append(
                    f"❌ {info['display']} is missing!\n"
                    f"   Purpose: {info['purpose']}\n"
                    f"   Get one at: {info['how_to']}\n"
                    f"   Set it in .env: {key}=your-key-here"
                )
            elif status == APIKeyStatus.INVALID:
                errors.append(
                    f"⚠️  {info['display']} looks invalid: {preview}\n"
                    f"   Please update it in .env"
                )
        
        # Check optional keys
        for key, info in SurgeConfig.OPTIONAL_KEYS.items():
            status, preview = SurgeConfig.get_status(key)
            
            if status == APIKeyStatus.MISSING:
                warnings.append(
                    f"⚡ {info['display']} not configured\n"
                    f"   {info['note']}\n"
                    f"   Set it in .env if needed: {key}=your-key-here"
                )
        
        return len(errors) == 0, errors, warnings
    
    @staticmethod
    def print_config_report():
        """Print configuration report on startup"""
        print("\n" + "="*70)
        print("🚀 SURGE - PPTX Configuration Report")
        print("="*70 + "\n")
        
        # Check LLM provider
        llm_provider = os.getenv("LLM_PROVIDER", "google").upper()
        print(f"📌 LLM Provider: {llm_provider}")
        
        if llm_provider == "GOOGLE":
            model = os.getenv("GOOGLE_MODEL", "models/gemini-2.5-flash")
            print(f"   Model: {model}")
        
        print("\n📋 Required API Keys:\n")
        for key, info in SurgeConfig.REQUIRED_KEYS.items():
            status, preview = SurgeConfig.get_status(key)
            status_icon = status.value
            print(f"   {status_icon} {info['display']}")
            if status != APIKeyStatus.CONFIGURED:
                print(f"      Get one: {info['how_to']}")
        
        print("\n⚡ Optional API Keys:\n")
        for key, info in SurgeConfig.OPTIONAL_KEYS.items():
            status, preview = SurgeConfig.get_status(key)
            status_icon = status.value
            print(f"   {status_icon} {info['display']}")
            if status != APIKeyStatus.CONFIGURED:
                print(f"      {info['note']}")
                print(f"      Get one: {info['how_to']}")
        
        print("\n" + "-"*70 + "\n")
        
        is_valid, errors, warnings = SurgeConfig.validate_setup()
        
        if errors:
            print("❌ SETUP ERRORS:\n")
            for error in errors:
                print(f"   {error}\n")
        
        if warnings:
            print("⚡ SETUP WARNINGS:\n")
            for warning in warnings:
                print(f"   {warning}\n")
        
        if is_valid and not warnings:
            print("✅ All required API keys are configured!\n")
        elif is_valid:
            print("✅ Required API keys are configured (see warnings above)\n")
        
        print("-"*70)
        print("Rate Limiting: " + ("✅ Enabled" if os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true" else "❌ Disabled"))
        print(f"Rate Limit: {os.getenv('RATE_LIMIT_CALLS', '100')} calls per {os.getenv('RATE_LIMIT_WINDOW', '3600')}s")
        print("="*70 + "\n")
        
        return is_valid


def setup_config_logging():
    """Configure logging for config validation"""
    logger.info("🚀 Surge - PPTX Starting Up...")
    
    is_valid = SurgeConfig.print_config_report()
    
    if not is_valid:
        logger.warning("⚠️  Some required API keys are missing. Please configure them in .env")
    
    return is_valid
