import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

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

# Inicializar y entrenar el modelo Naive Bayes
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Predecir las etiquetas para el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Encontrar el mejor y peor vino según la predicción del modelo
best_wine = wine_data.loc[y_pred.argmax()]
worst_wine = wine_data.loc[y_pred.argmin()]

print("El mejor vino predicho:")
print(best_wine)
print("\nEl peor vino predicho:")
print(worst_wine)

# Graficar las etiquetas predichas
plt.figure(figsize=(10, 6))

# Representar los vinos rojos en rojo y los blancos en negro
plt.scatter(X_test.loc[wine_data['type'] == 'red', 'alcohol'], X_test.loc[wine_data['type'] == 'red', 'volatile acidity'],
            color='red', label='Red Wine')
plt.scatter(X_test.loc[wine_data['type'] == 'white', 'alcohol'], X_test.loc[wine_data['type'] == 'white', 'volatile acidity'],
            color='black', label='White Wine')

plt.title('Predicted Wine Quality')
plt.xlabel('Alcohol')
plt.ylabel('Volatile Acidity')
plt.legend()
plt.show()
