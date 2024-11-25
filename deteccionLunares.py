import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
import uvicorn

#evitar logs innecesarios de tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprime los logs de nivel INFO

app = FastAPI ()

model = load_model('/content/drive/MyDrive/imagenesLunares/melanoma.keras')

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis = 0)
    img = img / 255.0
    prediction = model.predict(img)
    prob_benigno = prediction [0][0]
    prob_maligno = prediction[0][1]

    return prob_benigno, prob_maligno

@app.post("/predict/")
async def predict(file: UploadFile = File ('...')):
    img_path = f"temp_{file.file_name}"
    with open (img_path, "wb") as f:
        f.write(await file.read())

        prob_benigno, prob_maligno = predict_image(img_path)
        os.remove(img_path)

        return {
            "Probabilidad de que sea benigno": f"{prob_benigno * 100:.2f}%",
            "Probabilidad de que sea maligno": f"{prob_maligno * 100:.2f}%"
        }

# probar en render
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)