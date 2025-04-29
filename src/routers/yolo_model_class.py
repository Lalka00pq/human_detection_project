# python
import shutil
import io
import os
# project
from src.schemas.service_output import InferenceResult, Keypoints_yolo_models, FrameDetection
from src.tools.logging_tools import get_logger
# 3rd party
from ultralytics import YOLO
import torch
from PIL import Image
from fastapi import UploadFile, File
import cv2

logger = get_logger()


class ModelYolo:
    """Класс моделей YOLO"""

    def __init__(self, model_path: str,
                 model_type: str,
                 confidence: float,
                 device: str = 'cpu',
                 ) -> None:
        """Инициализация модели YOLO

        Args:
            model_path (str, optional): Путь до модели.
            device (str, optional): Устройство для выполнения детекции('cuda' или 'cpu').
            model_type (str, optional): Формат модели. 
            confidence (float, optional): Уверенность в детекции.
        """
        self.device = device
        self.model_name = model_path.split('/')[-1]
        self.model_path = model_path + '.' + model_type
        self.model = YOLO(self.model_path)
        self.model_type = model_type
        self.confidence = confidence
        logger.info(
            f"Модель {self.model_name} загружена и используется на устройстве {self.device}"
        )

    def set_model_confidence(self, confidence: float) -> None:
        """Устанавливает уровень уверенности модели

        Args:
            confidence (float): Уверенность модели

        Returns:
            None
        """
        self.model.conf = confidence
        logger.info(
            f"Уровень уверенности модели {self.model_name} установлен на {confidence}"
        )

    def change_device(self, device: str = 'cpu') -> None:
        """Метод для изменения устройства модели

        Args:
            device (str, optional): Устройство для выполнения детекции('cuda' или 'cpu'). 
        Returns:
            None

        Raises:
            ValueError: Если устройство не поддерживается 
        """
        if device == 'cuda' and torch.cuda.is_available():
            self.device = device
        elif device == 'cuda' and not torch.cuda.is_available():
            logger.info(
                "CUDA не доступна на устройстве. Используется CPU для выполнения детекции объектов."
            )
            self.device = 'cpu'
        elif device == 'cpu':
            self.device = device
        else:
            raise ValueError(
                f"Устройство {device} не поддерживается. Используйте 'cuda' или 'cpu'")
        logger.info(
            f"Модель {self.model_name} переведена на устройство {device}"
        )

    def predict(self, image: File, conf: float = 0.25) -> YOLO:
        """Метод для предсказания модели

        Args:
            image_path (File): Загруженный файл

        Returns:
            Keypoints_yolo_models: Ключевые точки модели
        """
        image_for_detect = Image.open(
            io.BytesIO(image.file.read())).convert('RGB')
        if self.model_type == 'onnx':
            results = self.model(
                image_for_detect, device=self.device, conf=conf, verbose=False)
        elif self.model_type == 'pt':
            results = self.model.predict(
                source=image_for_detect, save=False, conf=conf, verbose=False, device=self.device)
        return results

    def load_video(self, video: UploadFile) -> str:
        """Метод для загрузки видео
        Args:
            video (UploadFile): Видео файл

        Returns:
            str: Путь к видеофайлу
        """
        video_path = f"temp_{video.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        return video_path

    def video_detection(self, path_to_video: str) -> list | None:
        """Метод для детекции объектов на видео

        Args:
            path_to_video (str): Путь к видео

        Returns:
            list | None: Список объектов, обнаруженных на видео
        """
        # frame_skip = 5
        class_names = ['Standing', 'Lying']
        cap = cv2.VideoCapture(path_to_video)
        if not cap.isOpened():
            logger.error("Не удалось открыть видеофайл")
            return None
        frames = []
        frame_id = 0
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_FPS, 10)
            ret, frame = cap.read()
            if not ret:
                break
            # if frame_id % frame_skip != 0:
            #     frame_id += 1
            #     continue
            detection = self.model.predict(
                frame, device=self.device, conf=self.confidence, verbose=False)
            frame_result = []
            for row in detection:
                boxes = row.boxes
                keypoints = row.keypoints
                # ids = row.boxes.id
                for i in range(len(boxes)):
                    box = boxes[i]
                    # object_id = ids[i]
                    xyxy = box.xyxy[0].tolist()
                    xmin, ymin, xmax, ymax = xyxy
                    cls_obj = box.cls[0].item()
                    class_name = class_names[int(cls_obj)]
                    current_keypoints = keypoints[i].xy[0].tolist()
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
                    frame_result.append(InferenceResult(
                        class_name=class_name,
                        # track_id=int(object_id),
                        x=int(xmin + (xmax - xmin) / 2),
                        y=int(ymin + (ymax - ymin) / 2),
                        width=int(xmax - xmin),
                        height=int(ymax - ymin),
                        keypoints=keypoints_yolo
                    ))
                    # logger.info(
                    #     f"Объект {class_name} c id  обнаружен на изображении с координатами: ({xmin}, {ymin}), ({xmax}, {ymax}),\
                    #           с вероятностью {box.conf[0].item()}"
                    # )
            frames.append(FrameDetection(
                frame=frame_id, detections=frame_result))
            logger.info(
                f"Кадр {frame_id} обработан. Объектов на кадре: {len(frame_result)}"
            )
            frame_id += 1
        cap.release()
        os.remove(path_to_video)
        logger.info(
            "Видео обработано"
        )
        return frames

    def get_points(self, results: YOLO) -> list | None:
        """Метод для получения ключевых точек из результатов детекции

        Args:
            results (YOLO): Результаты детекции на изображении

        Returns:
            list | None: Список объектов, обнаруженных на изображении
        """
        class_names = ['Standing', 'Lying']
        detected_objects = []
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            for i in range(len(boxes)):
                box = boxes[i]
                xyxy = box.xyxy[0].tolist()
                xmin, ymin, xmax, ymax = xyxy
                cls_obj = box.cls[0].item()
                class_name = class_names[int(cls_obj)]
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
                    f"Состояние {class_name}. Координатами: ({xmin}, {ymin}), ({xmax}, {ymax}),\
                          Вероятностью {box.conf[0].item()}"
                )
        if len(detected_objects) == 0:
            logger.info(
                "Объекты на изображении не обнаружены"
            )
            return None

        return detected_objects
