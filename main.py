import json
from typing import Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException


from worker_api import NeuralWorker

worker = NeuralWorker(model_path="best.pt")
app = FastAPI()


def clean_text(obj: Any, chars_to_remove: str = '/,') -> Any:
    if isinstance(obj, dict):
        return {key: clean_text(value, chars_to_remove) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_text(item, chars_to_remove) for item in obj]
    elif isinstance(obj, str):

        for char in chars_to_remove:
            obj = obj.replace(char, '')
        obj = ' '.join(obj.split())
        return obj.strip()
    else:
        return obj

@app.post("/start_neuro")
async def neuro_api(
        class_name: str = Form(...,description="Пример: {"'"participant_name"'": {"'"text"'": "'"(Зжcюже | Coemvoa"'"}}"),
        image: UploadFile = File(...),
) -> Any:
    try:
        class_dict = json.loads(class_name)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    contents = await image.read()

    result = worker.process_image(contents)
    result_null = clean_text(result)
    compare, accuracy = worker.compare_text(result_null, class_dict)

    return result, class_dict, compare, accuracy, result_null


if __name__ == '__main__':
    uvicorn.run(app,host="0.0.0.0", port=8000)
    # uvicorn.run(app, port=8000)