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
    classes: list = Field(default=["human",
                                   "wind/sup-board",
                                   "boat",
                                   "bouy",
                                   "sailboat",
                                   "kayak"])
    """Список классов"""


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


class FrameDetection(BaseModel):
    """Модель для результата инференса кадра"""
    frame: int
    """Номер кадра"""
    detections: List[InferenceResult]
    """Детекции на кадре"""


class DetectedAndClassifiedObject(BaseModel):
    """ Датакласс данных которые будут возвращены сервисом (детекция и классификация) """
    object_bbox: List[InferenceResult] | None
    """ Координаты объекта """


class DetectionAndClassificationVideodataOutput(BaseModel):
    """Датаконтракт выхода сервиса"""
    objects: List[FrameDetection]
    """Список объектов"""
