import gradio as gr
import numpy as np
import pandas as pd
import joblib
import json

# FUNCIONES DEL MODELO DE REGRESIÓN LOGÍSTICA (necesarias para la predicción)
def funcion_sigmoide(z):
    z_clipped = np.clip(z, -500, 500) # Prevenir overflow/underflow
    g = 1 / (1 + np.exp(-z_clipped))
    return g

def funcion_hipotesis_logistica(X_matrix_b, theta_vector):
    if theta_vector.ndim == 1: # Asegurar que theta es un vector columna
        theta_vector = theta_vector.reshape(-1, 1)
    z = X_matrix_b @ theta_vector # Multiplicación de matrices
    return funcion_sigmoide(z)
