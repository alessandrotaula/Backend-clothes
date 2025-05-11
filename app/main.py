from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uuid
from app.classify import process_image  # importa la tua funzione
import shutil
import os

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Salva temporaneamente l'immagine
    img_id = str(uuid.uuid4())
    img_path = f"temp/{img_id}.jpg"
    os.makedirs("temp", exist_ok=True)
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Esegui classificazione
    df = process_image(img_path)

    # Salva immagini delle maschere
    result = []
    for i, row in df.iterrows():
        mask_path = f"temp/{img_id}_{i}.png"
        row["image"].save(mask_path)
        result.append({
            "id": i,
            "category": row["category"],
            "color_name": row["color_name"],
            "mask_path": mask_path
        })

    return JSONResponse(content=result)
