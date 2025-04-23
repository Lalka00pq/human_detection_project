# python
import json
# 3rdparty
import numpy as np
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from pydantic import TypeAdapter

import torch
from torchvision import transforms


# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger
from src.schemas.service_output import FrameDetection, DetectionVideodataOutput
from src.routers.api_interface_image.router import ModelYolo
logger = get_logger()

service_config = r".\src\configs\service_config.json"

with open(service_config, "r") as json_service_config:
    service_config_dict = json.load(json_service_config)

logger.info(f"Конфигурация сервиса: {service_config}")

service_config_adapter = TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(
    service_config_dict)

router = APIRouter(tags=["Detection Interfaces"], prefix="")


@router.post(
    "/video_inference",
    summary="Выполнение детекции на изображении"
)
async def inference(
        model_path: str = service_config_python.detectors_params.detector_model_path,
        model_type: str = service_config_python.detectors_params.detector_model_format,
        confidence: float = service_config_python.detectors_params.confidence_thershold,
        video: UploadFile = File(...),
) -> DetectionVideodataOutput | None:
    """Выполнение детекции на видео

    Args:
        model_path (str, optional): Путь к модели. Defaults to service_config_python.detectors_params.detector_model_path.
        model_type (str, optional): Формат модели (pt или onnx). Defaults to service_config_python.detectors_params.detector_model_format.
        confidence (float, optional): Уверенность при детекции. Defaults to service_config_python.detectors_params.confidence_thershold.
        video (UploadFile, optional): Видео. Defaults to File(...).

    Returns:
        DetectedAndClassifiedObject | None: _description_
    """
    model = ModelYolo(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_type=model_type,
        confidence=confidence,
    )
    # model.change_device(device=model.device)
    video_path = model.load_video(video=video)
    logger.info(f"Путь к видео: {video_path}")
    frames = model.video_detection(path_to_video=video_path)
    logger.info(f"Количество кадров: {len(frames)}")
    return DetectionVideodataOutput(objects=frames)
