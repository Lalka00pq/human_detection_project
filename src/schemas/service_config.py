# python
import json
from pathlib import Path
# 3rdparty
from pydantic import BaseModel, Field


class LoggingParams(BaseModel):
    """Датакласс, описывающий настройки логирования"""

    save_logs: bool = Field(default=True)
    """Сохранять ли логи работы сервиса"""
    logs_directory: str = Field(default=r".\src\logs")
    """Директория, в которую предполагается сохранять логи работы сервиса"""
    logging_config: str = Field(default=r".\src\logging.yaml")
    """Путь к YAML-конфигурации логирования"""


class CommonParams(BaseModel):
    """Датакласс, описывающий общие настройки сервиса"""

    host: str = Field(default="localhost")
    """Адрес хоста сервиса"""
    port: int = Field(default=8000)
    """Порт сервиса"""


class DetectorParams(BaseModel):
    """Датакласс, описывающий параметры детектора"""
    detector_name: str = Field(
        default="yolo11x-pose-100epochs")

    detector_model_format: str = Field(default="pt")
    detector_model_path: str = Field(
        default="./src/models/detectors/trained models/yolo11m-pose-100epochs")
    confidence_threshold: float = Field(default=0.25)
    nms_threshold: float = Field(default=0.5)
    use_cuda: bool = Field(default=False)


class ClassesInfo(BaseModel):
    """Датакласс, описывающий названия классов"""
    classes_name: list = Field(
        default=["Standing", "Lying"])


class ServiceConfig(BaseModel):
    """Конфигурация сервиса"""
    detectors_params: DetectorParams = Field(default=DetectorParams())
    """Параметры детектора"""
    classes_info: ClassesInfo = Field(default=ClassesInfo())
    """Названия классов"""
    logging_params: LoggingParams = Field(default=LoggingParams())
    """Параметры логирования"""
    common_params: CommonParams = Field(default=CommonParams())
    """Общие настройки сервиса (хост, порт)"""

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> "ServiceConfig":
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(
            detectors_params=DetectorParams(**data['detector_params']),
            classes_info=ClassesInfo(**data['classes_info']),
            logging_params=LoggingParams(**data['logging_params']),
            common_params=CommonParams(**data['common_params']),
        )
