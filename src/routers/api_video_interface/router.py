# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger
from src.schemas.service_output import DetectionVideodataOutput
# 3rdparty
from fastapi import APIRouter, File, UploadFile, Request
from concurrent.futures import ThreadPoolExecutor
import asyncio
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
        model_path (str): Путь к модели.
        model_type (str): Формат модели (pt или onnx).
        confidence (float): Уверенность при детекции.
        use_cuda (bool): Использовать ли GPU.
        video (UploadFile): Видео для детекции.

    Returns:
        DetectedAndClassifiedObject | None: Pydantic модель объектов, обнаруженных на изображении.
    """
    model = request.app.state.model
    if model is None:
        logger.error("Модель не загружена")
        return None
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
