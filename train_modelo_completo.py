import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Conexión a MySQL
try:
    conn = mysql.connector.connect(
        host="localhost", user="root", password="", database="traductor_gestos"
    )
    print("Conectado a MySQL para entrenamiento...")

    # Leemos la tabla 'gestos'
    df = pd.read_sql("SELECT * FROM gestos", conn)
    conn.close()
except Exception as e:
    print(f"Error al conectar con la base de datos: {e}")
    exit()

if df.empty:
    print("La base de datos está vacía. Captura datos con 'captura_sql.py' primero.")
    exit()

# 2. Preparación de datos
# Quitamos 'id' y 'letra' para que queden solo las coordenadas X, Y, Z
X = df.drop(columns=["id", "letra"])
y = df["letra"]

# Dividimos para pruebas (80% entrenamiento, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenamiento del Modelo
print(f"Entrenando con {len(df)} registros encontrados...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train.values, y_train)  # Usamos .values para evitar avisos de nombres de columna

# 4. Evaluación rápida
y_pred = rf.predict(X_test.values)
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.2%}")

# 5. Guardar el modelo que usará tu MAIN
joblib.dump(rf, "modelo_final.pkl")
print("¡Cerebro actualizado! Archivo 'modelo_final.pkl' generado exitosamente.")