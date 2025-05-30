import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import joblib

df = pd.read_csv('StudentPerformanceFactors.csv')

numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 
               'Tutoring_Sessions', 'Physical_Activity', 'Previous_Scores']
categorical_cols = ['Parental_Involvement', 'Access_to_Resources', 
                   'Extracurricular_Activities', 'Motivation_Level',
                   'Internet_Access', 'Teacher_Quality', 'School_Type', 
                   'Peer_Influence', 'Learning_Disabilities',
                   'Parental_Education_Level', 'Distance_from_Home', 'Gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def evaluate_model(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'Dataset': set_name,
        'MSE': round(mse, 2),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R-squared': round(r2, 4)
    }
    return metrics

train_metrics = evaluate_model(y_train, y_train_pred, 'Training')
test_metrics = evaluate_model(y_test, y_test_pred, 'Test')

results_df = pd.DataFrame([train_metrics, test_metrics])
print("\n" + "="*50)
print("EVALUACIÓN DEL MODELO DE REGRESIÓN")
print("="*50)
print(results_df.to_string(index=False))
print("\n" + "="*50)

print("\nINTERPRETACIÓN CLAVE:")
print(f"- Error promedio (MAE): {test_metrics['MAE']:.1f} puntos")
print(f"- Margen de error típico (RMSE): ±{test_metrics['RMSE']:.1f} puntos")
print(f"- % de varianza explicada (R²): {test_metrics['R-squared']*100:.1f}%")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Predicciones vs Valores Reales (Test Set)')
plt.xlabel('Calificación Real')
plt.ylabel('Predicción del Modelo')
plt.grid(True)
plt.show()

joblib.dump(model, 'newModel.pkl')