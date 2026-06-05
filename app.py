import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="IA: Detección de Frustración", layout="wide")
st.title("🧠 Sistema de Reconocimiento de Frustración en Contextos de Autismo")
st.write("Proyecto de Innovación Tecnológica - Inteligencia Artificial 1")

# Intentar cargar localmente, si no, pedir archivo
modelo = None
nombre_modelo = 'modelo_frustracion_autismo.h5'

if os.path.exists(nombre_modelo):
    try:
        modelo = tf.keras.models.load_model(nombre_modelo)
        st.success("¡Modelo cargado exitosamente desde el sistema local!")
    except Exception as e:
        st.error(f"Error al leer el archivo local: {e}")
else:
    st.warning("⚠️ No se encontró el archivo del modelo en la carpeta del proyecto.")
    archivo_subido = st.file_uploader("Por favor, arrastra y suelta aquí tu archivo 'modelo_frustracion_autismo.h5'", type=["h5"])
    if archivo_subido is not None:
        with open(nombre_modelo, "wb") as f:
            f.write(archivo_subido.getbuffer())
        modelo = tf.keras.models.load_model(nombre_modelo)
        st.success("¡Modelo subido e instalado correctamente!")
        st.rerun()

# Carga de la matriz de confusión en la barra lateral
st.sidebar.header("📊 Rendimiento Académico")
st.sidebar.metric(label="Precisión General (Accuracy)", value="71.0%", delta="Métricas Balanceadas")

nombre_imagen = "matriz_confusion_campeona.png"
if os.path.exists(nombre_imagen):
    st.sidebar.image(nombre_imagen, caption="Matriz de Confusión del Proyecto")
else:
    imagen_subida = st.sidebar.file_uploader("Subir imagen de la matriz (Opcional)", type=["png", "jpg"])
    if imagen_subida is not None:
        with open(nombre_imagen, "wb") as f:
            f.write(imagen_subida.getbuffer())
        st.rerun()

# Control de la cámara en vivo
if modelo is not None:
    comenzar_demo = st.checkbox("⚙️ Activar Cámara Web en Tiempo Real")

    if comenzar_demo:
        caja_video = st.image([]) 
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened() and comenzar_demo:
            ret, frame = cap.read()
            if not ret:
                st.warning("No se pudo acceder a la cámara web.")
                break
            
            frame = cv2.flip(frame, 1)
            h_original, w_original, _ = frame.shape
            
            # PREPROCESAMIENTO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_redimensionada = cv2.resize(rgb_frame, (224, 224))
            img_normalizada = img_redimensionada / 255.0
            vector_entrada = np.expand_dims(img_normalizada, axis=0)
            
            # PREDICCIÓN
            prediccion = modelo.predict(vector_entrada, verbose=0)
            clase_resultado = np.argmax(prediccion)
            
            # CORRECCIÓN DE FORMATO NUMÉRICAMENTE SEGURO
            probabilidad = float(prediccion[0][clase_resultado] * 100)
            
            # ALERTAS VISUALES
            if clase_resultado == 1:
                estado_texto = f"ALERTA: Frustración / Estrés Detectado ({probabilidad:.1f}%)"
                color_banner = (255, 0, 0)
            else:
                estado_texto = f"Estado: Calma / Estable ({probabilidad:.1f}%)"
                color_banner = (0, 255, 0)
                
            cv2.rectangle(frame, (0, 0), (w_original, 60), color_banner, -1)
            cv2.putText(frame, estado_texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            caja_video.image(frame, channels="BGR")
            
        cap.release()
else:
    st.info("💡 El sistema activará los controles de la cámara en cuanto se cargue el archivo del modelo arriba.")
