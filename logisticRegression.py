#Regresion logistica hecha con el framework de SciKit Learn
#Creado por José Pablo Cruz Ramos  Matricula: A01138740

#Para correr el programa se necesitan de las siguientes librerias
#favor de instalarlas si se corren de manera local con el comando pip install
import numpy as np # comanddos para matematicas y estadistica
import pandas as pd # procesamiento de los datos con la funcion .read_csv entre otras funciones
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#se da uso del datosframe otorgado en clases llamado iris.csv 
data_iris = pd.read_csv("iris.csv")
dummy_lbl = LabelEncoder()
print(" ")
#visualizamos el dataframe con sus primeros 5 registros
print(" ")
print(data_iris.head())
print(" ")
#Para poder obtener un mejor modelo, debemos asegurarnos de que los registros de nuestro dataframe non cuente con atipicoss, valores nulos o duplicados
print("Limpieza del data frame de IRIS \n")
valores_nulos = data_iris.isna().sum().sum()
print("Valores nulos encontrados: " + str(valores_nulos))
valores_duplicados = data_iris.duplicated().sum()
print("Valores duplicados encontrados: " + str(valores_duplicados) + "\n")
print("Pasando a variable dummy la especia de las plantas\n")
data_iris["species"] = dummy_lbl.fit_transform(data_iris["Species"])
print(data_iris.head())
# se sabe que la variable de iris de la especie es una variable categorica y no una cualitativa, si lo que buscamos es predecir dicha variable entonces debemos 
#convertirla en una variable dummy, usaremos 0,1,2 para poder usar las 3 especies presentes en el dataframe

#Ahora que tenemos nuestros datos en orden y transformados, podemos usar el modelo de regresion logistica
data_iris_aux = pd.read_csv("iris.csv")
data_iris_aux["species"] = dummy_lbl.fit_transform(data_iris["Species"])
data_iris_aux.drop(["Species"], axis=1, inplace=True)
x = data_iris_aux.drop(["species"],axis=1) #ingresamos las variables utilizadas para predecir la variable objetivo en una variable
y = data_iris["species"] #pasamos nuestra variable objetivo "y" a una variable de lista

regresionLog = LogisticRegression(fit_intercept=True, penalty='l2', tol=1e-5, C=0.8, solver='lbfgs', max_iter=600)
x_entren, x_prueba, y_entren, y_prueba = train_test_split(x, y, test_size = 0.33, random_state = 42, stratify=y) #con la funcion de train_test_split generamos nuestras muestras 
#de datos a utilizar tanto para el entrenamiento del modelo de regresion logistica como la muestra de prueba para la comparacion y verificacion de precision del modelo

#Usaremos una funcion de standarscaler con la finalidad de ajustar los datos continuos a que se comporten con una distribucion normal, de esta manera ayudamos al modelo
# de regresion logistica a que obtenga mejores resultados
estandarizador = StandardScaler()
x_entren_std = estandarizador.fit_transform(x_entren)
x_prueba_std = estandarizador.transform(x_prueba)
#ahora que tenemos nuestro set de entrenamiento podemos entrenar el modelo 
regresionLog.fit(x_entren_std,y_entren)
predicciones = regresionLog.predict(x_prueba_std)
#para poder observar las metricas y precision del modelo al estimar daremos uso de una matriz de confusion y del valor de accuracy score de scikit learn
puntaje = accuracy_score(y_prueba, predicciones)
print("\n")
print("Acurracy score: " + str(puntaje))
print("Matriz de confusion \n")
print(confusion_matrix(y_prueba,predicciones))
print("\n")
#Tambien se encontro esta metrica de sci kit learn la cual se encarga de generar un reporte de la clasificacion del modelo y que tan precisa es
print("Report de clasificacion de especies:")
print(classification_report(y_prueba, predicciones))
