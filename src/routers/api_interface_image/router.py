# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger
from src.schemas.service_output import DetectedAndClassifiedObject
from src.routers.yolo_model_class import ModelYolo
# 3rdparty
import numpy as np
from fastapi import APIRouter, File, UploadFile
from PIL import Image

import torch
from torchvision import transforms

logger = get_logger()

service_config_python = ServiceConfig.from_json_file(
    '.\src\configs\service_config.json')

router = APIRouter(tags=["Detection Inferences"], prefix="")


def preprocess_image(image_path: str) -> np.ndarray:
    """Предобработка изображения для классификатора

    Args:
        image_path (str): Путь к изображению

    Returns:
        np.ndarray: массив изображения
    """
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch.numpy()


@router.post(
    "/image_inference",
    summary="Выполняет инференс изображения."
)
async def inference(
        model_path: str = service_config_python.detectors_params.detector_model_path,
        model_type: str = service_config_python.detectors_params.detector_model_format,
        confidence: float = service_config_python.detectors_params.confidence_thershold,
        image: UploadFile = File(...),
) -> DetectedAndClassifiedObject | None:
    """Метод для инференса изображения

    Args:
        model_path (str): Путь до модели.
        model_type (str): Формат модели.
        confidence (float): Уверенность в детекции.
        image (UploadFile): Изображение для детекции. 

    Returns:
        DetectedAndClassifiedObject | None: Pydantic модель объектов, обнаруженных на изображении.
    """
    model = ModelYolo(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_type=model_type,
        confidence=confidence,
    )
    # model.change_device(device=model.device)

    results = model.predict(image=image, conf=confidence)

    detected_objects = model.get_points(results=results)

    if detected_objects is None:
        return DetectedAndClassifiedObject(object_bbox=None)
    return DetectedAndClassifiedObject(object_bbox=detected_objects)
