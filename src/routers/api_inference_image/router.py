# python
import time
# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger
from src.schemas.service_output import DetectedAndClassifiedObject
from src.routers.api_check_model_loaded.router import check_model_loaded
from src.routers.api_background_description.router import get_background_description
# 3rdparty
from fastapi import APIRouter, File, UploadFile, Request

logger = get_logger()

service_config_python = ServiceConfig.from_json_file(
    r'.\src\configs\service_config.json')

router = APIRouter(tags=["Detection Inferences"], prefix="")


@router.post(
    "/image_inference",
    summary="Выполняет инференс изображения."
)
async def inference(
        request: Request,
        use_cuda: bool = service_config_python.detectors_params.use_cuda,
        image: UploadFile = File(...),
) -> DetectedAndClassifiedObject | None:
    """Метод для детекции объектов на изображении.

    Args:
        model_path (str): Путь до модели.
        model_type (str): Формат модели.
        confidence (float): Уверенность в детекции.
        use_cuda (bool): Использовать ли GPU.
        image (UploadFile): Изображение для детекции. 

    Returns:
        DetectedAndClassifiedObject | None: Pydantic модель объектов, обнаруженных на изображении.
    """
    model_check = await check_model_loaded(request)
    if model_check is False:
        logger.info("Модель не загружена")
        return DetectedAndClassifiedObject(object_bbox=None)
    model = request.app.state.model
    logger.info(f"Используется модель {model.model_name}")
    if use_cuda:
        model.change_device(
            device='cuda')
    start = time.time()
    results = model.predict(image=image, conf=model.confidence, iou=model.iou)

    detected_objects = model.get_points(results=results)
    end = time.time()
    logger.info(f"Время выполнения инференса: {end - start}")
    response = get_background_description(image)
    logger.info(response)
    if detected_objects is None:
        logger.info("Объекты не обнаружены")
        return DetectedAndClassifiedObject(object_bbox=None)
    return DetectedAndClassifiedObject(object_bbox=detected_objects)
