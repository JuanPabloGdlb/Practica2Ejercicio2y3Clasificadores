import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos
red_wine_data = pd.read_csv("winequality-red.csv", sep=';')
white_wine_data = pd.read_csv("winequality-white.csv", sep=';')

# Agregar una columna 'type' para distinguir entre vinos rojos y blancos
red_wine_data['type'] = 'red'
white_wine_data['type'] = 'white'

# Combinar los dos conjuntos de datos
wine_data = pd.concat([red_wine_data, white_wine_data], ignore_index=True)

# Separar características (X) y etiquetas (y)
X = wine_data.drop(['quality', 'type'], axis=1)
y = wine_data['quality']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializar los modelos
models = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "K-Vecinos Cercanos": KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
    "Máquinas de Soporte Vectorial": SVC(kernel='rbf', gamma='scale'),
    "Naive Bayes": GaussianNB()
}

# Calcular métricas para cada modelo
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    sensitivity = recall_score(y_test, y_pred, average='weighted', zero_division=0)  # Sensitivity is the same as Recall
    cm = confusion_matrix(y_test, y_pred)
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    if (true_negatives + false_positives) > 0:
        specificity = true_negatives / (true_negatives + false_positives)
    else:
        specificity = np.nan
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1 Score": f1
    }

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(10, 6))

# Ocultar ejes
ax.axis('off')

# Crear tabla y mostrar
table = ax.table(cellText=results_df.values, colLabels=results_df.columns, rowLabels=results_df.index, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Método ganador
winner = results_df.idxmax(axis=1).iloc[0]
plt.text(0.5, -0.1, f"El método ganador es: {winner}", ha='center', fontsize=12, transform=ax.transAxes)

# Ajustar diseño de la figura
plt.tight_layout()

plt.show()
