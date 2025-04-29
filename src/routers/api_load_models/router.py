from fastapi import APIRouter, Request
from src.tools.logging_tools import get_logger
from src.routers.yolo_model_class import ModelYolo
import time
from src.schemas.service_config import ServiceConfig
router = APIRouter(tags=["Load Models"], prefix="")
logger = get_logger()
service_config_python = ServiceConfig.from_json_file(
    r'.\src\configs\service_config.json')


@router.post(
    "/load_model",
    summary="Load a model from a file"
)
async def load_model(
    request: Request,
    model_path: str = service_config_python.detectors_params.detector_model_path,
    model_type: str = service_config_python.detectors_params.detector_model_format,
    confidence: float = service_config_python.detectors_params.confidence_threshold,
):
    """ Метод для загрузки модели из файла.

    Args:
        model_path (str): Путь до модели
        model_type (str): Тип модели
        confidence (float): Уверенность модели в детекции

    Returns:
        json: Запись о статусе загрузки модели
    """
    start = time.time()
    model = ModelYolo(
        model_path=model_path,
        device='cpu',
        model_type=model_type,
        confidence=confidence,
    )
    end = time.time()
    logger.info(f"Модель загружена за {end - start} секунд")
    request.app.state.model = model
    return {'model': 'Модель загружена'}
