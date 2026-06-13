# NumPy se utiliza para trabajar con arreglos y datos numéricos
import numpy as np

# Matplotlib permite crear gráficas para visualizar los resultados
import matplotlib.pyplot as plt

# KMeans es el algoritmo de clustering que usaremos para agrupar clientes
from sklearn.cluster import KMeans



# DATOS DE LOS CLIENTES

# Cada fila representa un cliente.
# El primer valor corresponde al número de compras realizadas al mes.
# El segundo valor corresponde al gasto mensual en dólares.
X = np.array([
    [2, 150],
    [1, 120],
    [3, 180],
    [2, 140],
    [15, 900],
    [18, 1100],
    [17, 980],
    [16, 1020],
    [8, 450],
    [9, 500],
    [7, 420]

])



# CREACIÓN DEL MODELO

# Se crea un modelo K-Means con 3 grupos (clusters).
# random_state se utiliza para obtener siempre el mismo resultado
# al ejecutar el programa varias veces.
modelo = KMeans(n_clusters=3, random_state=42)



# ENTRENAMIENTO DEL MODELO

# El algoritmo analiza los datos y busca patrones para formar grupos.
modelo.fit(X)



# OBTENER LOS RESULTADOS

# labels_ indica a qué grupo pertenece cada cliente.
etiquetas = modelo.labels_

# cluster_centers_ almacena las coordenadas de los centroides,
# es decir, el centro de cada grupo encontrado.
centroides = modelo.cluster_centers_



# MOSTRAR RESULTADOS EN CONSOLA

print("Clasificación de clientes:\n")

# enumerate permite obtener el índice y los datos del cliente.
for i, cliente in enumerate(X):

    # cliente[0] = número de compras
    # cliente[1] = gasto mensual
    # etiquetas[i] = grupo asignado por K-Means
    print(
        f"Cliente {i + 1}: "
        f"Compras = {cliente[0]}, "
        f"Gasto = ${cliente[1]} "
        f"--> Grupo {etiquetas[i]}"
    )



# GRAFICAR LOS CLIENTES

# Se crea un diagrama de dispersión (scatter plot).
# X[:, 0] toma todos los valores de la primera columna (compras).
# X[:, 1] toma todos los valores de la segunda columna (gasto).
# c=etiquetas asigna un color diferente a cada grupo.
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=etiquetas
)



# GRAFICAR LOS CENTROIDES

# Los centroides representan el centro de cada cluster.
# marker='X' utiliza una X para identificarlos.
# s=250 aumenta el tamaño del marcador.
plt.scatter(
    centroides[:, 0],
    centroides[:, 1],
    marker='X',
    s=250
)



# CONFIGURAR LA GRÁFICA

# Etiqueta del eje X.
plt.xlabel("Número de compras al mes")

# Etiqueta del eje Y.
plt.ylabel("Gasto mensual ($)")

# Título de la gráfica.
plt.title("Segmentación de clientes usando K-Means")



# MOSTRAR LA GRÁFICA

# Abre una ventana con la visualización final.
plt.show()