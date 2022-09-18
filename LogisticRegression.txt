# LogisticRegression

Proyecto para creacion de un modelo de regresion logistica para clasificacion de las especies de flores con base en sus datos cuantitativos.
Se dio uso del dataset de IRIS.csv otorgado durante el curso para diferentes actividades y asi mismo se utilizo el framework de SciKit Learn para la
implementaicon del modelo de regresion logistica.

En la primera seccion del codigo podemos ubicar las librerias que se deben instalar con el comando pip install para poder correr el programa localmente.
Despues pasamos a la limpieza y exploracion de los datos principalmente para ubicar aquellos datos que sean nulos, duplicados o incluso outliers.
Despues pasamos a hacer la columna de Species a una variable Dummy para poder utilizarla dentro del modelo como una variable cuantitativa.

Se dio uso de un plugin llamado Label Encoder para pasar la variable de especies de nombres a una escala de numeros (0,1,2), igualmente se importo un modulo llamado standarsScaler para poder aproximar los datos lo mas posible a una distribucion normal, con la finalidad de mejorar el aprendizaje del modelo logistico a crear. 

Para el modelo se crearon las muestras de prueba y entrenamiento con la funcino de train_test_split la cual permite incluso modificar la forma en la que dividie los datos aleatoriamente.Se generaron las predicciones y se utilizaron algunas metricas para comprobar la precision del modelo como la matriz de confusion, el accuracy score y el reporte de clasificacion, todos provienen del framework de Scikit Learn. 
