import requests, json, csv, time
from pathlib import Path

API_URL = "http://localhost:8000/start_neuro"
IMAGE_FOLDER = "worker/dataset_v2/images"
results = []
start_total = time.time()
count = 0
with open("worker/dataset_v2/labels.csv", "r", encoding="utf-8-sig") as f:
    for image_name, fio in csv.reader(f, delimiter=';'):
        image_name, fio = image_name.strip(), fio.strip()

        image_path = Path(IMAGE_FOLDER) / image_name

        print(f"\nОбработка: {fio}")
        print(f"Файл: {image_path}")

        if not image_path.exists():
            print(f"Файл не найден: {image_path}")
            continue

        start_img = time.time()

        with open(image_path, "rb") as img:
            resp = requests.post(API_URL,
                                 files={"image": (image_name, img, "image/png")},
                                 data={"class_name": json.dumps({"participant_name": {"text": fio}})})

        elapsed_img = time.time() - start_img

        if resp.status_code == 200:
            try:
                r = resp.json()
                print(r[0].get("detections", {}).get("participant_name"))
                results.append([fio, image_name, r[2], r[3], f"{elapsed_img:.2f}"])

                print(f"{fio}: совпадение={r[2]}, точность={r[3]}%, время={elapsed_img:.2f}с")
                if r[3] >= 100:
                    count += 1
                    print(count)
            except Exception as e:
                print(f"Ошибка: {e}")
        else:
            results.append([fio, image_name, "ERROR", 0, f"{elapsed_img:.2f}"])
            print(f"{fio}: ошибка, время={elapsed_img:.2f}с")

        time.sleep(0.5)

total_time = time.time() - start_total

print(f"\n{'=' * 50}")
print(f"Обработано {len(results)} изображений")
print(f"Количество 100%: {count}")
print(f"Общее время: {total_time:.2f} сек ({total_time / 60:.2f} мин)")
print(f"Среднее время: {total_time / len(results):.2f} сек/изображение")
print(f"{'=' * 50}")