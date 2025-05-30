import pandas as pd
import joblib

# 1. Cargar datos y modelo
df = pd.read_csv('StudentPerformanceFactors.csv')
model = joblib.load('modelo_calificaciones.pkl')

# 2. Extraer fila específica (ej: fila 5)
fila = df.iloc[1351].copy().drop('Exam_Score', errors='ignore')

print(fila.to_dict())
# 3. Convertir y predecir
input_data = pd.DataFrame([fila.to_dict()])
score = model.predict(input_data)[0]

print(f"Predicción para la fila 5: {score:.1f}/100")