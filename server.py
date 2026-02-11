from dotenv import load_dotenv
from pathlib import Path
import os
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
import uvicorn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port number to run the server on",
    )
    parser.add_argument(
        "--reload", type=str, default="false", help="Reload the server on code changes"
    )
    args = parser.parse_args()
    reload = args.reload.strip().lower() in {"true", "1", "yes", "on"}
    
    uvicorn.run(
        "api.main:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=args.port,
        log_level="info",
        reload=reload,
    )
