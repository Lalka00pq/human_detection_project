# python
from datetime import datetime
from typing import List

# 3rdparty
from pydantic import Field, BaseModel


class HealthCheck(BaseModel):
    """Датакласс для описания статуса работы нейросетевого сервиса"""

    status_code: int
    """Код статуса работы нейросетевого сервиса"""
    datetime: datetime
    """Отсечка даты и времени"""


class GetClassesOutput(BaseModel):
    """Датаконтракт выхода сервиса"""
    classes: list = Field(default=["Standing", "Falling"])
    """Список классов"""


class Keypoints_yolo_models(BaseModel):
    """Модель для ключевых точек"""
    nose: List[float] | None = None
    """Координаты носа"""
    left_eye: List[float] | None = None
    """Координаты левого глаза"""
    right_eye: List[float] | None = None
    """Координаты правого глаза"""
    left_ear: List[float] | None = None
    """Координаты левого уха"""
    right_ear: List[float] | None = None
    """Координаты правого уха"""
    left_shoulder: List[float] | None = None
    """Координаты левого плеча"""
    right_shoulder: List[float] | None = None
    """Координаты правого плеча"""
    left_elbow: List[float] | None = None
    """Координаты левого локтя"""
    right_elbow: List[float] | None = None
    """Координаты правого локтя"""
    left_wrist: List[float] | None = None
    """Координаты левого запястья"""
    right_wrist: List[float] | None = None
    """Координаты правого запястья"""
    left_hip: List[float] | None = None
    """Координаты левого бедра"""
    right_hip: List[float] | None = None
    """Координаты правого бедра"""
    left_knee: List[float] | None = None
    """Координаты левого колена"""
    right_knee: List[float] | None = None
    """Координаты правого колена"""
    left_ankle: List[float] | None = None
    """Координаты левой лодыжки"""
    right_ankle: List[float] | None = None
    """Координаты правой лодыжки"""


class InferenceResult(BaseModel):
    """Модель для результата инференса изображения"""
    class_name: str
    """Имя класса"""
    x: int
    """Координата x"""
    y: int
    """Координата y"""
    width: int
    """Ширина"""
    height: int
    """Высота"""
    keypoints: Keypoints_yolo_models | None = None
    """Координаты ключевых точек"""


# Классы для работы с изображением
# --------------------------------
class DetectedAndClassifiedObject(BaseModel):
    """ Датакласс данных которые будут возвращены сервисом (детекция и классификация) """
    object_bbox: List[InferenceResult] | None
    """ Координаты объекта """


# Классы для работы с видео
# --------------------------------
class FrameDetection(BaseModel):
    """Модель для результата инференса кадра"""
    frame: int
    """Номер кадра"""
    detections: List[InferenceResult]
    """Детекции на кадре"""


class DetectionAndClassificationVideodataOutput(BaseModel):
    """Датаконтракт выхода сервиса"""
    objects: List[FrameDetection]
    """Список объектов"""
