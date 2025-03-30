# python
import json
import io
# 3rdparty
import numpy as np
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from pydantic import TypeAdapter

import torch
from torchvision import transforms
import ultralytics


# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger
from src.schemas.service_output import InferenceResult, DetectedAndClassifiedObject
logger = get_logger()

service_config = r".\src\configs\service_config.json"

with open(service_config, "r") as json_service_config:
    service_config_dict = json.load(json_service_config)

logger.info(f"Конфигурация сервиса: {service_config}")

service_config_adapter = TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(
    service_config_dict)

router = APIRouter(tags=["Main FastAPI service router"], prefix="")


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
    "/inference",
    summary="Выполняет инференс изображения."
)
async def inference(

        image: UploadFile = File(...),
) -> DetectedAndClassifiedObject | None:
    """Метод для инференса изображения

    Args:
        image (UploadFile, optional): Изображение. Defaults to File(...).

    Returns:
        InferenceResult: Результат инференса
    """
    # Определение устройства (cuda или cpu) для выполнения инференса
    if torch.cuda.is_available():
        device = 'cuda'
    elif not torch.cuda.is_available():
        logger.info(
            "CUDA не доступна на устройстве. Используется CPU для выполнения инференса"
        )
        device = 'cpu'
    else:
        device = 'cpu'
    logger.info(
        f"Устройство для выполнения инференса - {device}"
    )
    detector_model = ultralytics.YOLO(
        'src/models/detectors/yolo11n-pose.pt').to(device)
    image_for_detect = Image.open(
        io.BytesIO(image.file.read())).convert('RGB')
    detected_objects = []
    
    results = detector_model.predict(source=image_for_detect)
    
    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
        for i in range(len(boxes)):
            box = boxes[i]
            xyxy = box.xyxy[0].tolist()
            xmin, ymin, xmax, ymax = xyxy
            cls_obj = box.cls[0].item()
            class_name = detector_model.names[int(cls_obj)]
            current_keypoints = keypoints[i].xy[0].tolist() if keypoints is not None else []
            detected_objects.append(InferenceResult(
                class_name=class_name,
                x=int(xmin + (xmax - xmin) / 2),
                y=int(ymin + (ymax - ymin) / 2),
                width=int(xmax - xmin),
                height=int(ymax - ymin),
                keypoints=current_keypoints,
            ))
    if len(detected_objects) == 0:
        logger.info(
            "Объекты на изображении не обнаружены"
        )
        return DetectedAndClassifiedObject(object_bbox=None)
    return DetectedAndClassifiedObject(object_bbox=detected_objects)
