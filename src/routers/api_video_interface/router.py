# project
from src.schemas.service_config import ServiceConfig
from src.tools.logging_tools import get_logger
from src.schemas.service_output import DetectionVideodataOutput
from src.routers.yolo_model_class import ModelYolo
# 3rdparty
from fastapi import APIRouter, File, UploadFile
logger = get_logger()

service_config_python = ServiceConfig.from_json_file(
    r'.\src\configs\service_config.json')

router = APIRouter(tags=["Detection Inferences"], prefix="")


@router.post(
    "/video_inference",
    summary="Выполнение детекции на изображении"
)
async def inference(
        model_path: str = service_config_python.detectors_params.detector_model_path,
        model_type: str = service_config_python.detectors_params.detector_model_format,
        confidence: float = service_config_python.detectors_params.confidence_thershold,
        use_cuda: bool = service_config_python.detectors_params.use_cuda,
        video: UploadFile = File(...),
) -> DetectionVideodataOutput | None:
    """Выполнение детекции на видео

    Args:
        model_path (str): Путь к модели.
        model_type (str): Формат модели (pt или onnx).
        confidence (float): Уверенность при детекции.
        use_cuda (bool): Использовать ли GPU.
        video (UploadFile): Видео для детекции.

    Returns:
        DetectedAndClassifiedObject | None: Pydantic модель объектов, обнаруженных на изображении.
    """
    model = ModelYolo(
        model_path=model_path,
        device='cpu',
        model_type=model_type,
        confidence=confidence,
    )
    if use_cuda:
        model.change_device(
            device='cuda')

    video_path = model.load_video(video=video)
    logger.info(f"Путь к видео: {video_path}")
    frames = model.video_detection(path_to_video=video_path)
    logger.info(f"Количество кадров: {len(frames)}")
    return DetectionVideodataOutput(objects=frames)
