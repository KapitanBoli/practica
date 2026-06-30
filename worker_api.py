from typing import Any, Tuple
from io import BytesIO
import torch
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import easyocr

import logging
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralWorker:
    def __init__(self, model_path: str = "best.pt", custom_ocr_path: str = "model/v4"):

        self.model_path = model_path
        self.custom_ocr_path = custom_ocr_path
        self.languages= ['ru']
        self.model = None
        self.reader = None
        self.gpu_available = torch.cuda.is_available()
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.warning(f"Доступность GPU: {self.gpu_available}")
            logger.info(f"Загрузка YOLO модели из {self.model_path}")
            self.model = YOLO(self.model_path)

            logger.info(f"Загрузка EasyOCR: языки={self.languages}")
            self.reader = easyocr.Reader(
                self.languages,
                user_network_directory = self.custom_ocr_path,
                model_storage_directory = self.custom_ocr_path,
                gpu=self.gpu_available,
                recog_network="ru_ocr"
            )

            logger.info("Модели успешно загружены")

        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            raise

    @staticmethod
    def _open_image(image) -> Any:
        if isinstance(image, bytes):
            image = BytesIO(image)
        with Image.open(image) as img_pil:
            img_pil = ImageOps.exif_transpose(img_pil)

            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')

            img_np = np.array(img_pil)
            return img_np, img_pil.size


    def _process_yolo(self, img_np) -> Any:
        try:
            return self.model.predict(source=img_np, imgsz=640, conf=0.25, verbose=False)
        except Exception as e:
            logger.error(f"Ошибка обработки YOLO: {e}")
            raise

    def _process_ocr(self, model_result, img_np, W, H) -> dict[str, Any]:
        image_result = {
            "detections": {}
        }
        for j, box in enumerate(model_result[0].boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls_id]

            x_center, y_center, w_norm, h_norm = box.xywhn[0].tolist()

            x1 = int((x_center - w_norm / 2) * W)
            y1 = int((y_center - h_norm / 2) * H)
            x2 = int((x_center + w_norm / 2) * W)
            y2 = int((y_center + h_norm / 2) * H)

            # Ограничение координат (защита)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            cropped = img_np[y1:y2, x1:x2]

            # OCR
            ocr_result = self.reader.readtext(cropped, detail=0)
            # ocr_text = " | ".join(ocr_result) if ocr_result else "[не найдено]"
            ocr_text = " ".join(ocr_result) if ocr_result else "[не найдено]"

            image_result["detections"][label] = {
                "confidence": round(conf, 4),
                "text": ocr_text
            }
        return image_result

    def process_image(self, image) -> dict[str, Any] | None:
        try:
            img_np, img_pil_size = self._open_image(image)
            result_yolo = self._process_yolo(img_np)
            if len(result_yolo[0].boxes) > 0:
                logger.info(f"Обнаружено объектов: {len(result_yolo[0].boxes)}")
                return self._process_ocr(result_yolo, img_np, *img_pil_size)
            else:
                 logger.info("Объекты не обнаружены")
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            return {"error": f"Ошибка обработки: {str(e)}"}
        finally:
            logger.info("Конец обработки изображения")

    @staticmethod
    def compare_text(data: dict[str, Any] | None, criteria: dict[str, dict]) -> Tuple[bool, float | None]:
        accuracy = 0.0
        if not data:
            logger.info(f"Сравнение: {False}")
            return False, accuracy

        detections = data.get("detections", {})

        for class_name, class_criteria in criteria.items():
            detection = detections.get(class_name)
            if not detection:
                logger.info(f"Класс {class_name} не найден")
                logger.info(f"Сравнение: {False}")
                return False, accuracy

            for field, expected_value in class_criteria.items():
                detected_value = detection.get(field, "")
                expected_clean = ' '.join(expected_value.lower().split())
                detected_clean = ' '.join(detected_value.lower().split())

                try:
                    accuracy = SequenceMatcher(None, detected_clean, expected_clean).ratio()
                    accuracy = round(accuracy * 100, 2)
                except Exception as e:
                    logger.error(f"Ошибка accuracy: {e}")

                # if detection.get(field) != expected_value:
                if expected_clean != detected_clean:
                    logger.info(f"Поле {field} для класса {class_name} не совпадает")
                    logger.info(f"Сравнение: {False}")
                    return False, accuracy


        logger.info(f"Сравнение: {True}")
        return True, accuracy


# if __name__ == "__main__":
#     worker = NeuralWorker()
#     result = worker.process_image("test/image.jpg")
#     print(result)
#     compare = worker.compare_text(result, {"participant_name": {"text": "(Зжcюже | Coemvoa"},
#     "doctype": {"text": "ДИПЛОМ"}})
#     print(compare)