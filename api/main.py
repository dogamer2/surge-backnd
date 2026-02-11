from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from api.lifespan import app_lifespan
from api.middlewares import UserConfigEnvUpdateMiddleware, RateLimitMiddleware
from api.v1.ppt.router import API_V1_PPT_ROUTER
from api.v1.webhook.router import API_V1_WEBHOOK_ROUTER
from api.v1.system import SYSTEM_ROUTER
from api.v1.mock.router import API_V1_MOCK_ROUTER
from utils.get_env import get_app_data_directory_env


app = FastAPI(lifespan=app_lifespan)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount app data files from configured app data directory
configured_app_data_dir = get_app_data_directory_env()
app_data_dir = (
    Path(configured_app_data_dir)
    if configured_app_data_dir
    else (Path(__file__).parent.parent / "app_data")
)
app_data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/app_data", StaticFiles(directory=str(app_data_dir)), name="app_data")

# Routers
app.include_router(API_V1_PPT_ROUTER)
app.include_router(API_V1_WEBHOOK_ROUTER)
app.include_router(SYSTEM_ROUTER)
app.include_router(API_V1_MOCK_ROUTER)

# Middlewares
origins = ["*"]

# Rate limiting middleware (added first so it runs after CORS)
app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(UserConfigEnvUpdateMiddleware)
