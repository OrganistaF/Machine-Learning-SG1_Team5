import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib

# Cargar el dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# 1. Definir nuevas columnas basadas en tu aclaración
numeric_cols = [
    'Hours_Studied',
    'Attendance',        # Asumiremos que es numérica (ej: porcentaje 0-100)
    'Sleep_Hours',
    'Tutoring_Sessions',
    'Physical_Activity',
    'Previous_Scores'    # Agregado por lógica de predicción
]

categorical_cols = [
    'Parental_Involvement',
    'Access_to_Resources',  # Asumido como categórico (ej: 'High','Medium','Low')
    'Extracurricular_Activities',
    'Motivation_Level',
    'Internet_Access',
    'Teacher_Quality',
    'School_Type',
    'Peer_Influence',
    'Learning_Disabilities',
    'Parental_Education_Level',
    'Distance_from_Home',
    'Gender'
]

# 2. Preprocesamiento con pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 3. Crear modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])

# 4. Preparar datos
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenar y evaluar
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f'MAE Entrenamiento: {mean_absolute_error(y_train, train_pred):.2f}')
print(f'MAE Prueba: {mean_absolute_error(y_test, test_pred):.2f}')

# 6. Función de predicción
def predict_score(input_data):
    """Predice la calificación basada en características de entrada"""
    columns = [
        'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
        'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
        'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    input_df = pd.DataFrame([input_data], columns=columns)
    return model.predict(input_df)[0]

# 7. Ejemplo de uso
if __name__ == "__main__":
    user_input = {
        'Hours_Studied': 4.5,
        'Attendance': 95,  # Porcentaje
        'Parental_Involvement': 'High',
        'Access_to_Resources': 'Medium',
        'Extracurricular_Activities': 'Yes',
        'Sleep_Hours': 8,
        'Previous_Scores': 82,
        'Motivation_Level': 'High',
        'Internet_Access': 'Yes',
        'Tutoring_Sessions': 2,
        'Family_Income': 52000,
        'Teacher_Quality': 'Excellent',
        'School_Type': 'Private',
        'Peer_Influence': 'Positive',
        'Physical_Activity': 5,
        'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'Masters',
        'Distance_from_Home': 'Medium',
        'Gender': 'Female'
    }
    
    prediction = predict_score(user_input)
    print(f'\nPredicción de calificación: {prediction:.1f} puntos')
    
    joblib.dump(model, 'modelo_calificaciones.pkl')