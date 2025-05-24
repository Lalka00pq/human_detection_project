# python
import asyncio
import sys

# 3rdparty
from fastapi import FastAPI

# project
from src.routers.api_info import router as InfoRouter
from src.routers.api_inference_image import router as DetectAndClassifyRouter
from src.routers.api_get_classes_info import router as GetClassesInfoRouter
from src.routers.api_inference_video import router as VideoInterfaceRouter
from src.routers.api_check_model_loaded import router as CheckModelLoadedRouter
from src.routers.api_load_models.router import router as LoadModelsRouter
from src.routers.api_background_description.router import router as BackgroundDescriptionRouter
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(
    title="FastAPI human falling detection",
    version="0.1.0",
    description="Service for detection human standing or falling",
    docs_url=None,
    redoc_url=None,
)
api_v1_prefix = ""
app.state.model = None
app.include_router(InfoRouter, prefix=api_v1_prefix)
app.include_router(LoadModelsRouter, prefix=api_v1_prefix)
app.include_router(DetectAndClassifyRouter, prefix=api_v1_prefix)
app.include_router(GetClassesInfoRouter, prefix=api_v1_prefix)
app.include_router(VideoInterfaceRouter, prefix=api_v1_prefix)
app.include_router(CheckModelLoadedRouter, prefix=api_v1_prefix)
app.include_router(BackgroundDescriptionRouter, prefix=api_v1_prefix)
app.docs_url = "/docs"
app.redoc_url = "/redocs"
app.setup()
