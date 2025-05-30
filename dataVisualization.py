import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Configuración de visualización
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})
sns.set_palette("viridis")

# %% [code] ##########################
#### 2. CARGA Y EXPLORACIÓN DE DATOS ####
##########################
# Cargar datos
df = pd.read_csv('StudentPerformanceFactors.csv')

# Mostrar estructura básica
print("="*50)
print("Resumen de datos:")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print("="*50)
print(df.info())

# Estadísticas descriptivas
desc_stats = df.describe(include='all').T
desc_stats['missing'] = df.isnull().sum()
desc_stats['dtype'] = df.dtypes
desc_stats['unique'] = df.nunique()
print("\nEstadísticas descriptivas:")
print(desc_stats)

# %% [code] ##########################
#### 3. LIMPIEZA DE DATOS ####
##########################
# Manejo de valores faltantes
df_original = df.copy()
print("\nValores faltantes antes de limpieza:")
print(df.isnull().sum())

# Estrategias de imputación
df['Parental_Involvement'].fillna('Medium', inplace=True)
df['Family_Income'].fillna(df['Family_Income'].mode()[0], inplace=True)
df['Previous_Scores'].fillna(df['Previous_Scores'].median(), inplace=True)

# Manejo de outliers
df = df[(df['Hours_Studied'] <= 30) & 
        (df['Attendance'] >= 50) &
        (df['Sleep_Hours'] >= 4) &
        (df['Previous_Scores'] <= 100)]

print("\nValores faltantes después de limpieza:")
print(df.isnull().sum())
print(f"\nFilas eliminadas: {df_original.shape[0] - df.shape[0]} ({(df_original.shape[0] - df.shape[0])/df_original.shape[0]*100:.1f})")

# %% [code] ##########################
#### 4. ANÁLISIS Y VISUALIZACIÓN ####
##########################
# Distribución de la variable objetivo
plt.figure(figsize=(10, 6))
sns.histplot(df['Exam_Score'], kde=True, bins=20)
plt.title('Distribución de Calificaciones de Examen')
plt.xlabel('Calificación')
plt.ylabel('Frecuencia')
plt.axvline(df['Exam_Score'].mean(), color='r', linestyle='--', label=f'Media: {df["Exam_Score"].mean():.1f}')
plt.legend()
plt.savefig('distribucion_calificaciones.png', bbox_inches='tight')
plt.show()

# Correlaciones numéricas
numeric_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 
                'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
            annot_kws={"size": 12}, vmin=-1, vmax=1)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.savefig('correlaciones_numericas.png', bbox_inches='tight')
plt.show()

# Relación entre variables clave
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Calificación vs Notas Previas
sns.scatterplot(x='Previous_Scores', y='Exam_Score', hue='Teacher_Quality', 
                data=df, ax=axes[0, 0], s=80, alpha=0.7)
axes[0, 0].set_title('Calificación vs Notas Previas')
axes[0, 0].set_xlabel('Notas Previas')
axes[0, 0].set_ylabel('Calificación Examen')

# Calificación vs Horas de Estudio
sns.boxplot(x=pd.cut(df['Hours_Studied'], bins=5), y='Exam_Score', 
            data=df, ax=axes[0, 1])
axes[0, 1].set_title('Calificación por Horas de Estudio')
axes[0, 1].set_xlabel('Horas de Estudio (bins)')
axes[0, 1].set_ylabel('Calificación Examen')
axes[0, 1].tick_params(axis='x', rotation=45)

# Impacto de Factores Institucionales
sns.barplot(x='School_Type', y='Exam_Score', hue='Internet_Access', 
            data=df, ax=axes[1, 0], ci=None)
axes[1, 0].set_title('Impacto de Tipo de Escuela y Acceso a Internet')
axes[1, 0].set_xlabel('Tipo de Escuela')
axes[1, 0].set_ylabel('Calificación Promedio')

# Impacto de Factores Personales
sns.violinplot(x='Motivation_Level', y='Exam_Score', hue='Parental_Involvement', 
               data=df, split=True, inner="quart", ax=axes[1, 1])
axes[1, 1].set_title('Impacto de Motivación y Participación Parental')
axes[1, 1].set_xlabel('Nivel de Motivación')
axes[1, 1].set_ylabel('Calificación Examen')

plt.tight_layout()
plt.savefig('relaciones_clave.png', bbox_inches='tight')
plt.show()

df['Study_Efficiency'] = df['Previous_Scores'] / (df['Hours_Studied'] + 1)
df['Resources_Access'] = df['Access_to_Resources'].map({'Low': 0, 'Medium': 1, 'High': 2})

categorical_cols = ['Parental_Involvement', 'Extracurricular_Activities', 
                    'Motivation_Level', 'Internet_Access', 'Teacher_Quality',
                    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                    'Parental_Education_Level', 'Distance_from_Home', 'Gender']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop('Exam_Score', axis=1)
y = df_encoded['Exam_Score']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTamaño de conjuntos: Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")