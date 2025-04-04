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


class Keypoint(BaseModel):
    """Модель для ключевых точек"""
    pass


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
    keypoints: List[List[float]] | None = None
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
