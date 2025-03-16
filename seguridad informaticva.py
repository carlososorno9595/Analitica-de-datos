import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # Para balancear clases

# 📌 Cargar datos
ruta_archivo = "C:/Users/TI724/Documents/SCRIPS/datos_seguridad_informatica.csv"
df = pd.read_csv(ruta_archivo, delimiter=";")

# 📌 Procesamiento de datos
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')  
df = df.dropna(subset=['Fecha'])  
df['Año'] = df['Fecha'].dt.year
df['Mes'] = df['Fecha'].dt.month
df['Día'] = df['Fecha'].dt.day
df['Hora'] = df['Fecha'].dt.hour
df.drop(columns=['Fecha'], inplace=True)

# 📌 Codificación de variables categóricas
encoder = LabelEncoder()
columnas_categoricas = ['Tipo_Evento', 'IP_Origen', 'Usuario_Afectado', 'Estado', 'Nivel_Riesgo']
for col in columnas_categoricas:
    df[col] = encoder.fit_transform(df[col])

# 📌 Variables predictoras y objetivo
X = df.drop(columns=['Nivel_Riesgo'])
y = df['Nivel_Riesgo']

# 📌 Balanceo de Clases con SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# 📌 División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Optimización con GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                           param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 📌 Mejor Modelo Encontrado
best_model = grid_search.best_estimator_

# 📌 Predicciones
y_pred = best_model.predict(X_test)

# 📌 Evaluación
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print(f"✅ Precisión del modelo mejorado: {accuracy:.2f}")
print("\n📊 Reporte de clasificación:\n", report)
print("\n🔎 Matriz de confusión:\n", matrix)

