import gradio as gr
import numpy as np
import pandas as pd
import joblib # Para cargar el scaler
import json

def funcion_sigmoide(z):
    """Calcula la función sigmoide, evitando overflow/underflow."""
    z_clipped = np.clip(z, -500, 500) 
    g = 1 / (1 + np.exp(-z_clipped))
    return g

def funcion_hipotesis_logistica(X_matrix_b, theta_vector):
    """Calcula las probabilidades predichas por la regresión logística."""
    if theta_vector.ndim == 1: # Asegurar que theta es un vector columna
        theta_vector = theta_vector.reshape(-1, 1)
    z = X_matrix_b @ theta_vector # Multiplicación de matrices: X_matrix_b * theta_vector
    return funcion_sigmoide(z)
# --- Modelo de Regresión Logística (Donación de Sangre) ---
theta_logistic = None
scaler_logistic = None
feature_names_logistic = None
try:
    theta_logistic = np.load('theta_logistic.npy')
    scaler_logistic = joblib.load('scaler_logistic.joblib')
    with open('feature_names_logistic.json', 'r') as f:
        feature_names_logistic = json.load(f)
    print("Artefactos del modelo logístico cargados exitosamente.")
except FileNotFoundError:
    print("ERROR CRÍTICO: Faltan archivos para el modelo logístico (theta_logistic.npy, scaler_logistic.joblib, o feature_names_logistic.json).")
    print("Asegúrate de haberlos guardado desde tu notebook y que estén en el mismo directorio que app.py.")
except Exception as e:
    print(f"Error al cargar artefactos logísticos: {e}")
