import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de objetos")
image = Image.open('imagenes.jpg')
st.image(image, width=350)

with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Run the inference
    prediction = model.predict(data)
    print(prediction)

    # Mapeo de las 4 clases según tus requerimientos
    if prediction[0][0] > 0.5:
        st.header('Hola Humano, con Probabilidad: ' + str(prediction[0][0]))
    elif prediction[0][1] > 0.5:
        st.header('Celular, con Probabilidad: ' + str(prediction[0][1]))
    elif prediction[0][2] > 0.5:
        st.header('Audifonos, con Probabilidad: ' + str(prediction[0][2]))
    elif prediction[0][3] > 0.5:
        st.header('No veo nada, con Probabilidad: ' + str(prediction[0][3]))
    else:
        st.header('No hay suficiente certeza en la predicción')

