import pandas as pd
import joblib

df = pd.read_csv('../Data/StudentPerformanceFactors.csv')
model = joblib.load('model.pkl')

fila = df.iloc[7].copy().drop('Exam_Score', errors='ignore')

print(fila.to_dict())

input_data = pd.DataFrame([fila.to_dict()])
score = model.predict(input_data)[0]

print(f"Predicción para la fila 5: {score:.1f}/100")