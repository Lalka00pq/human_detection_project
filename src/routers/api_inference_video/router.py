# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger
from src.schemas.service_output import DetectionVideodataOutput
from src.routers.api_check_model_loaded.router import check_model_loaded
# 3rdparty
from fastapi import APIRouter, File, UploadFile, Request
logger = get_logger()

service_config_python = ServiceConfig.from_json_file(
    r'.\src\configs\service_config.json')

router = APIRouter(tags=["Detection Inferences"], prefix="")


@router.post(
    "/video_inference",
    summary="Выполнение детекции на изображении"
)
async def inference(
        request: Request,
        use_cuda: bool = service_config_python.detectors_params.use_cuda,
        video: UploadFile = File(...),
) -> DetectionVideodataOutput | None:
    """Выполнение детекции на видео

    Args:
        use_cuda (bool): Использовать ли GPU.
        video (UploadFile): Видео для детекции.

    Returns:
        DetectedAndClassifiedObject | None: Pydantic модель объектов, обнаруженных на изображении.
    """
    model_check = await check_model_loaded(request)
    if model_check is False:
        logger.info("Модель не загружена")
        return None
    model = request.app.state.model
    logger.info(f"Используется модель {model.model_name}")
    if use_cuda:
        model.change_device(
            device='cuda')

    video_path = model.load_video(video=video)
    logger.info(f"Путь к видео: {video_path}")
    # frames = await asyncio.to_thread(model.video_detection, video_path)
    frames = model.video_detection(path_to_video=video_path)
    logger.info(f"Количество кадров: {len(frames)}")
    return DetectionVideodataOutput(objects=frames)
