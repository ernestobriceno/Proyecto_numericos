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

# --- Modelo de Regresión Lineal (Ganancias de Empresas) ---
theta_lineal = None
mean_train_lineal = None
std_train_lineal = None
feature_names_lineal = None
try:
    theta_lineal = np.load('theta_lineal.npy')
    mean_train_lineal = np.load('mean_train_lineal.npy')
    std_train_lineal = np.load('std_train_lineal.npy')
    with open('feature_names_lineal.json', 'r') as f:
        feature_names_lineal = json.load(f)
    print("Artefactos del modelo lineal (normalización manual) cargados exitosamente.")
except FileNotFoundError:
    print("ERROR CRÍTICO: Faltan archivos para el modelo lineal (theta_lineal.npy, mean_train_lineal.npy, std_train_lineal.npy, o feature_names_lineal.json).")
    print("Asegúrate de haberlos guardado desde tu notebook y que estén en el mismo directorio que app.py.")
except Exception as e:
    print(f"Error al cargar artefactos lineales: {e}")
# --- Para Regresión Logística ---
def predecir_donacion_logistica(recencia, frecuencia, tiempo):
    """Realiza una predicción de donación de sangre."""
    if theta_logistic is None or scaler_logistic is None or feature_names_logistic is None:
        return "Error: Los artefactos del modelo logístico no se cargaron correctamente. Revisa la consola."
    try:
        # 1. Crear DataFrame con las entradas del usuario en el orden correcto
        # feature_names_logistic debe ser ['Recency_months', 'Frequency_times', 'Time_months']
        input_data = pd.DataFrame([[recencia, frecuencia, tiempo]], columns=feature_names_logistic)
        
        # 2. Escalar las características
        input_features_scaled = scaler_logistic.transform(input_data)
        
        # 3. Añadir el término de intercepto (columna de unos al principio)
        input_features_b = np.c_[np.ones((input_features_scaled.shape[0], 1)), input_features_scaled]
        
        # 4. Realizar la predicción de probabilidad
        probabilidad = funcion_hipotesis_logistica(input_features_b, theta_logistic)
        probabilidad_si_dona = probabilidad[0,0] # Extraer el valor escalar
        
        # 5. Convertir probabilidad a clase
        clase_predicha = 1 if probabilidad_si_dona >= 0.5 else 0
        resultado_clase = "Sí Donará (Clase 1)" if clase_predicha == 1 else "No Donará (Clase 0)"
        
        return f"{resultado_clase}\nProbabilidad de que sí done: {probabilidad_si_dona:.4f}"
    except Exception as e:
        return f"Error durante la predicción logística: {str(e)}"
# --- Para Regresión Lineal ---
def predecir_profit_lineal(rd_spend, administration, marketing_spend, state_input):
    """Realiza una predicción de profit de empresa."""
    if theta_lineal is None or mean_train_lineal is None or std_train_lineal is None or feature_names_lineal is None:
        return "Error: Los artefactos del modelo lineal no se cargaron correctamente. Revisa la consola."
    try:
        # 1. Crear un diccionario para las entradas numéricas y las dummies de estado
        # feature_names_lineal debe ser la lista de columnas usada para calcular mean_train_lineal y std_train_lineal
        # Ejemplo: ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']
        
        data_dict = {}
        # Llenar con las features numéricas
        data_dict['R&D Spend'] = rd_spend
        data_dict['Administration'] = administration
        data_dict['Marketing Spend'] = marketing_spend
        
        # Inicializar y establecer las variables dummy de estado
        # Asume que 'California' fue el estado base (drop_first=True)
        if 'State_Florida' in feature_names_lineal:
            data_dict['State_Florida'] = 1.0 if state_input == "Florida" else 0.0
        if 'State_New York' in feature_names_lineal:
            data_dict['State_New York'] = 1.0 if state_input == "New York" else 0.0
        # Si hay otras columnas dummy de estado en feature_names_lineal, inicialízalas a 0
        for col_name in feature_names_lineal:
            if col_name.startswith('State_') and col_name not in data_dict:
                data_dict[col_name] = 0.0

        # 2. Crear un DataFrame con las entradas en el ORDEN EXACTO de feature_names_lineal
        input_df_lineal = pd.DataFrame([data_dict])[feature_names_lineal]
        
        # 3. Normalización manual Z-score
        input_features_lineal_normalized_values = (input_df_lineal.values - mean_train_lineal) / std_train_lineal
        
        # 4. Añadir el término de intercepto
        input_features_lineal_b = np.c_[np.ones((input_features_lineal_normalized_values.shape[0], 1)), input_features_lineal_normalized_values]
        
        # 5. Realizar la predicción de profit (función de hipótesis lineal: h_theta(X) = X @ theta)
        profit_predicho = (input_features_lineal_b @ theta_lineal)[0,0]
        
        return f"Profit Predicho (Lineal): ${profit_predicho:,.2f}"
        
    except Exception as e:
        return f"Error durante la predicción lineal: {str(e)}"
