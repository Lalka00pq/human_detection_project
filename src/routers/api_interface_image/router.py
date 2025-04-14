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
from src.schemas.service_output import InferenceResult, DetectedAndClassifiedObject, Keypoints_yolo_models
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


class ModelYolo:
    def __init__(self, model_path: str = 'src/models/detectors/trained models/yolo11n-pose',
                 device: str = 'cpu',
                 model_type: str = 'pt') -> None:
        self.device = device
        self.model_name = model_path.split('/')[-1]
        self.model_path = model_path + '.' + model_type
        self.model = ultralytics.YOLO(self.model_path)
        self.model_type = model_type
        logger.info(
            f"Модель {self.model_name} загружена и используется на устройстве {self.device}"
        )

    def change_device(self, device: str = 'cpu') -> None:
        if device == 'cuda' and torch.cuda.is_available():
            self.device = device
            self.model = self.model.to(device)
        elif device == 'cuda' and not torch.cuda.is_available():
            logger.info(
                "CUDA не доступна на устройстве. Используется CPU для выполнения инференса"
            )
            self.device = 'cpu'
            self.model = self.model.to(device)
        elif device == 'cpu':
            self.device = device
            self.model = self.model.to(device)
        else:
            raise ValueError(
                f"Устройство {device} не поддерживается. Используйте 'cuda' или 'cpu'")
        logger.info(
            f"Модель {self.model_name} переведена на устройство {device}"
        )

    def predict(self, image: File):
        """Метод для предсказания модели

        Args:
            image_path (File): Загруженный файл

        Returns:
            Keypoints_yolo_models: Ключевые точки модели
        """
        image_for_detect = Image.open(
            io.BytesIO(image.file.read())).convert('RGB')
        if self.model_type == 'onnx':
            results = self.model(image_for_detect, device=self.device)
        elif self.model_type == 'pt':
            results = self.model.predict(source=image_for_detect, save=False)
        return results

    def get_points(self, results: ultralytics.YOLO) -> list | None:
        detected_objects = []
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            for i in range(len(boxes)):
                box = boxes[i]
                xyxy = box.xyxy[0].tolist()
                xmin, ymin, xmax, ymax = xyxy
                cls_obj = box.cls[0].item()
                class_name = self.model.names[int(cls_obj)]
                current_keypoints = keypoints[i].xy[0].tolist()
                # TODO: Выглядит колхозно, но работает. Нужно сделать лучше
                keypoints_yolo = Keypoints_yolo_models(
                    nose=current_keypoints[0],
                    left_eye=current_keypoints[1],
                    right_eye=current_keypoints[2],
                    left_ear=current_keypoints[3],
                    right_ear=current_keypoints[4],
                    left_shoulder=current_keypoints[5],
                    right_shoulder=current_keypoints[6],
                    left_elbow=current_keypoints[7],
                    right_elbow=current_keypoints[8],
                    left_wrist=current_keypoints[9],
                    right_wrist=current_keypoints[10],
                    left_hip=current_keypoints[11],
                    right_hip=current_keypoints[12],
                    left_knee=current_keypoints[13],
                    right_knee=current_keypoints[14],
                    left_ankle=current_keypoints[15],
                    right_ankle=current_keypoints[16],
                )
                detected_objects.append(InferenceResult(
                    class_name=class_name,
                    x=int(xmin + (xmax - xmin) / 2),
                    y=int(ymin + (ymax - ymin) / 2),
                    width=int(xmax - xmin),
                    height=int(ymax - ymin),
                    keypoints=keypoints_yolo,
                ))
                logger.info(
                    f"Объект {class_name} обнаружен на изображении с координатами: ({xmin}, {ymin}), ({xmax}, {ymax})"
                )
        if len(detected_objects) == 0:
            logger.info(
                "Объекты на изображении не обнаружены"
            )
            return None

        return detected_objects


@router.post(
    "/inference",
    summary="Выполняет инференс изображения."
)
async def inference(
        model_path: str = service_config_python.detectors_params.detector_model_path,
        model_type: str = service_config_python.detectors_params.detector_model_format,
        image: UploadFile = File(...),
) -> DetectedAndClassifiedObject | None:
    """Метод для инференса изображения

    Args:
        image (UploadFile, optional): Изображение. Defaults to File(...).

    Returns:
        InferenceResult: Результат инференса
    """
    model = ModelYolo(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_type=model_type
    )
    # model.change_device(device=model.device)

    results = model.predict(image=image)

    detected_objects = model.get_points(results=results)
    if detected_objects is None:
        return DetectedAndClassifiedObject(object_bbox=None)
    return DetectedAndClassifiedObject(object_bbox=detected_objects)

    # TODO: добавить возможность выбора формата модели
    # TODO: добавить возможность выбора конфигурации модели
