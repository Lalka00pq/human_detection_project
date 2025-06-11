# python
import time
# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger
from src.schemas.service_output import DetectedAndClassifiedObject
from src.routers.api_check_model_loaded.router import check_model_loaded
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
    start = time.time()
    model_check = await check_model_loaded(request)
    if model_check is False:
        logger.info("Модель не загружена")
        return DetectedAndClassifiedObject(object_bbox=None)
    model = request.app.state.model
    logger.info(f"Используется модель {model.model_name}")
    if use_cuda:
        model.change_device(
            device='cuda')
    results, width, height = model.predict(image=image, conf=model.confidence, iou=model.iou)
    detected_objects = model.get_points(results=results)
    need_check = False
    min_width = width * 0.2
    min_height = height * 0.2
    max_width = width * 0.8
    max_height = height * 0.8
    if detected_objects is not None:
        for obj in detected_objects:
            w, h = obj.width, obj.height
            keypoints = obj.keypoints
            if w < min_width or h < min_height or w > max_width or h > max_height:
                need_check = True
                break
            if keypoints and obj.class_name == "Standing":
                if keypoints.left_shoulder and keypoints.right_shoulder and keypoints.left_hip and keypoints.right_hip:
                    shoulder_y = (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) / 2
                    hip_y = (keypoints.left_hip[1] + keypoints.right_hip[1]) / 2
                    if abs(shoulder_y - hip_y) < min_height:
                        need_check = True
                        break
            elif keypoints and obj.class_name == "Lying":
                if keypoints.left_shoulder and keypoints.right_shoulder and keypoints.left_hip and keypoints.right_hip:
                    shoulder_x = (keypoints.left_shoulder[0] + keypoints.right_shoulder[0]) / 2
                    hip_x = (keypoints.left_hip[0] + keypoints.right_hip[0]) / 2
                    if abs(shoulder_x - hip_x) < min_width:  
                        need_check = True
                        break
    else:
        if model.confidence < 0.6:
            need_check = True
    end = time.time()
    
    logger.info(f"Время выполнения инференса: {end - start}")
    if detected_objects is None:
        logger.info("Объекты не обнаружены")
        return DetectedAndClassifiedObject(object_bbox=None, check_image=need_check)
    return DetectedAndClassifiedObject(object_bbox=detected_objects, check_image=need_check)
