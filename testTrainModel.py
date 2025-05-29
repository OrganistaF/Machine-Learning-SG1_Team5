import pandas as pd
import joblib

input_ejemplo = {
    # Numéricas
    'Hours_Studied': 4.5,               
    'Attendance': 90,                   
    'Sleep_Hours': 7.5,                 
    'Tutoring_Sessions': 2,             
    'Physical_Activity': 3,             
    'Previous_Scores': 85,              
    'Parental_Involvement': 'Medium',      
    'Access_to_Resources': 'High',         
    'Extracurricular_Activities': 'Yes',   
    'Motivation_Level': 'High',            
    'Internet_Access': 'Yes',              
    'Teacher_Quality': 'Good',             
    'School_Type': 'Public',               
    'Peer_Influence': 'Positive',          
    'Learning_Disabilities': 'No',         
    'Parental_Education_Level': 'Bachelor',
    'Distance_from_Home': 'Close',         
    'Gender': 'Female',                    
    'Family_Income': 45000                 
}

# 1. Cargar modelo (si lo guardaste)
modelo = joblib.load('modelo_calificaciones.pkl')

# 2. Predecir
prediccion = modelo.predict(pd.DataFrame([input_ejemplo]))
print(f"Calificación predicha: {prediccion[0]:.1f}/100")