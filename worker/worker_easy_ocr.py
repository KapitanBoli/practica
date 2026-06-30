import torch

import os
import json
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr

model_path = "../best.pt"

model = YOLO(model_path)

reader = easyocr.Reader(['en', 'ru'], gpu=torch.cuda.is_available())

test_folder = "test"

image_paths = [
    os.path.join(test_folder, f)
    for f in os.listdir(test_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"Будет обработано изображений: {len(image_paths)}")

output_dir = "test_results_yolo" #TODO: Убрать
output_json_dir = "test_results_yolo/json"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_json_dir, exist_ok=True)

all_results = []

for i, image_path in enumerate(image_paths):

    print("\n" + "=" * 70)
    print(f"Обработка {i + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

    image_result = {
        "filename": os.path.basename(image_path),
        "detections": []
    }

    try:
        with Image.open(image_path) as img_pil:
            img_pil = ImageOps.exif_transpose(img_pil)
            img_np = np.array(img_pil)
            W, H = img_pil.size

            results = model.predict(source=img_np, imgsz=640, conf=0.25, verbose=False)

            plt.figure(figsize=(10, 10))
            plt.imshow(img_pil)
            plt.axis('off')

            if len(results[0].boxes) > 0:

                print(f"Обнаружено объектов: {len(results[0].boxes)}")

                for j, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

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
                    ocr_result = reader.readtext(cropped, detail=0)
                    ocr_text = " | ".join(ocr_result) if ocr_result else "[не найдено]"

                    print(f"\n  {j + 1}. Класс: {label.upper()}")
                    print(f"     Confidence: {conf:.3f}")
                    print(f"     OCR текст: {ocr_text}")

                    image_result["detections"].append({
                        "class": label,
                        "confidence": round(conf, 4),
                        "text": ocr_text
                    })

                    # Рисуем рамку
                    plt.plot([x1, x2, x2, x1, x1],
                             [y1, y1, y2, y2, y1],
                             'r-', linewidth=2)

                    plt.text(x1, y1 - 10,
                             f'{label} {conf:.2f}',
                             bbox=dict(facecolor='red', alpha=0.5),
                             fontsize=12,
                             color='white')

            else:
                print("Объекты не обнаружены")

            save_path = os.path.join(output_dir, f"ocr_{os.path.basename(image_path)}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.show()
            plt.close()

            print(f"Сохранено изображение: {save_path}")

            all_results.append(image_result)

    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {str(e)}")
        continue

json_path = os.path.join(output_json_dir, "ocr_results.json")

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print("\n" + "=" * 70)
print(f"Обработано изображений: {len(image_paths)}")
print(f"JSON сохранён: {json_path}")
