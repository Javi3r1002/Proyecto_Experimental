#!/usr/bin/env python
# coding: utf-8

# # Proyecto de física experimental: Comparación de algoritmos de clasificación no supervisados aplicados al set de datos 17 de Sloan Digital Sky Survey (SDSS)
# 
# ## Universidad del Valle de Guatemala
# ### Javier Alejandro Mejía Alecio (20304)


#Se importan las librerías a utilizar
import numpy as np #Numpy es utilizado para las operaciones matemáticas
import pandas as pd #Pandas es utilizado para la manipulación de los datos 
import matplotlib.pyplot as plt # Matplotlib permite graficar los datos
import seaborn as sns # En este caso es utilizando para el gráfico de pairplot
from sklearn.model_selection import train_test_split # Nos permite separar los datos en un conjunto de entrenamiento y otro de prueba
from sklearn import datasets # Permite trabajar con grupos de información
from sklearn.tree import DecisionTreeClassifier #Algoritmo de arboles de decisión 
from sklearn import tree # Permite graficar el árbol
from sklearn import metrics #Metricas para medir el desempeño del algoritmo
from statsmodels.stats.outliers_influence import variance_inflation_factor #Librería que permite determinar el valor del VIF para detectar colinealidad entre variables
from sklearn.metrics import confusion_matrix #Metrica de matriz de confusión que permite visualizar los falos positivos y falsos negativos
from sklearn.neighbors import LocalOutlierFactor #Librería que permite la detección de puntos atípicos en la variables utilizadas
from sklearn.ensemble import RandomForestClassifier #Algortimo de clasificación Random Forest
from sklearn.datasets import make_classification #Permite hacer las clasificaciones
from sklearn.tree import export_graphviz #Librería enfocada a graficar los árboles
import random #En este caso la librería es utilizada por su función de seed que permite repetir la aleatoriedad de un caso
import timeit #Librería utilizada par medir el tiempo de ejecución
from sklearn.naive_bayes import GaussianNB #Algoritmo de clasificación Gausiano

Data = pd.read_csv('Skyserver_Data.csv') #Se importan las mediciones realizadas


print(Data.shape) #Se visualizan las dimensiones del conjunto de datos


Data = Data.replace(['STAR','GALAXY','QSO'], [0,1,2]) #Para poder aplicar los algoritmos se reemplazan los valores de 'star', 'Galaxy', 'Qso' por los valores 0,1,2 respectivamente



#Se toma una muestra de 50,000 mediciones

Muestra = Data.sample(50000, random_state = 123)

#Para poder manipular los datos de la muestra se reinicia el índice, es decir, que se inicia desde 0 a 49999

Muestra = Muestra.reset_index()


# ### Eliminación de puntos atípicos


#Se invoca el algortimo encargado de la detección de puntos atípicos en las variables que se tienen
clf = LocalOutlierFactor()
#Se le envían los datos al algoritmo para entrenarlo
y_pred = clf.fit_predict(Muestra)

x_score = clf.negative_outlier_factor_ #Se obtiene el punteo de punto atípico para cada una de las mediciones
# Se visualiza el punteo
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

#threshold
#Se define el umbral para el filtrado de datos atípicos
threshold = np.quantile(x_score , .10)                 
#Se obtiene el indice de las mediciones identificadas como puntos atípicos                           
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()


# In[10]:


print(outlier_index)


# In[11]:

#Se eliminan los puntos atípicos por los indices identificados anteriormente
Muestra.drop(outlier_index, inplace=True)

#Se obtiene las correlaciones lineal entre las variables del modelo y la variable objetivo, en este caso 'class'
correlation_mat = Muestra.corr()

#Se obtiene los valores de correlación lineal de las variables sobre la variable objetivo y se ordenan de menor a mayor

print(correlation_mat["class"].sort_values())


# Se toman las variables con una correlación mayor a |.1|

Muestra = Muestra[['mjd', 'plate', 'specobjid', 'i', 'u','r', 'g', 'redshift', 'class']]


#Utilizando Seaborn se obtienen la gráfica de pairplot
sns.pairplot(Muestra, hue="class",  diag_kws={'bw': .15})
plt.tight_layout()
plt.show()


#Se obtiene el valor de VIF para las variables del modelo
vif_data = pd.DataFrame() 
G = Muestra
vif_data["feature"] = G.columns
  
vif_data["VIF"] = [variance_inflation_factor(G.values, i)
                        #Se imprimen los valores calculados 
                          for i in range(len(G.columns))] 
  
print(vif_data)

#Se eliminan las variables con un valor de VIF mayor a 5
Muestra = Muestra.drop(['plate', 'specobjid'], axis = 1)


#Para poder utilizar los algoritmos se convierte la variable objetivo a tipo categoria
Muestra['class'] = Muestra['class'].astype('category')


y = Muestra.pop("class") #La variable respuesta se almacena en la variable 'y' y se elimina de la variable X para evitar problema de sobre ajuste
X = Muestra #El resto de los datos

random.seed(123)

# In[23]:

#Se separan los datos en datos de entrenamiento y prueba. Utilizando el 30 porciento de los datos como prueba y el restante 70 como entrenamiento
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7, random_state = 123)



# ### árbol de decisión

#Se invoca al algoritmo de clasificación y se define la profuncidad de este
arbol = DecisionTreeClassifier(max_depth=4, random_state=1) 
#Se entrena al algoritmo con los datos de entrenamiento
arbol = arbol.fit(X_train, y_train) 
#Se grafica el árbol
tree.plot_tree(arbol,feature_names= Muestra.columns, class_names=['0','1','2'],filled=True )
plt.title('árbol de decisión')
plt.show()
#Se 'clasifican' los datos de entrenamiento
y_predf = arbol.predict(X_train)


#Se clasifican los datos de prueba
y_pred = arbol.predict(X_test)
# Se obtienen las métricas del modelo de clasificación
print ("Accuracy entrenamiento para el algoritmo de árboles de decisión:",metrics.accuracy_score(y_train, y_predf))
print ("Accuracy para el algoritmo de árboles de decisión:",metrics.accuracy_score(y_test, y_pred))
print ("Precision para el algoritmo de árboles de decisión:", metrics.precision_score(y_test,y_pred,average='weighted') )
print ("Recall para el algoritmo de árboles de decisión: ", metrics.recall_score(y_test,y_pred,average='weighted'))

# Se obtiene la matriz de confusión
matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(matrix, annot=True, cbar=None, cmap="Blues")
plt.title("Matriz de confusión para el árbol de decisión"), plt.tight_layout()
plt.ylabel("Categoría real"), plt.xlabel("Predicted de la categoría")
plt.show()

# Random Forest:

RF = RandomForestClassifier(n_estimators=100,max_depth = 4, random_state = 1)
RF.fit(X_train, y_train)
plt.figure()
_ = tree.plot_tree(RF.estimators_[49], feature_names=X.columns, filled=True)
plt.title('Random Forest')
plt.show()

y_predf = RF.predict(X_train)


y_pred = RF.predict(X_test)
print ("Accuracy entrenamiento para el algoritmo Random Forest:",metrics.accuracy_score(y_train, y_predf))
print ("Accuracy para el algoritmo Random Forest:",metrics.accuracy_score(y_test, y_pred))
print ("Precision para el algoritmo Random Forest:", metrics.precision_score(y_test,y_pred,average='weighted') )
print ("Recall para el algoritmo Random Forest: ", metrics.recall_score(y_test,y_pred,average='weighted'))
matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(matrix, annot=True, cbar=None, cmap="Blues")
plt.title("Matriz de confusión para Random Forest"), plt.tight_layout()
plt.ylabel("Categoría real"), plt.xlabel("Predicted de la categoría")
plt.show()

#Naive Bayes


gaussian = GaussianNB()

gaussian.fit(X_train,y_train)
y_pred = gaussian.predict(X_test)
y_predT = gaussian.predict(X_train)
cm = confusion_matrix(y_test,y_pred)


# %%
accuracy_Entre = metrics.accuracy_score(y_train, y_predT)
accuracy=metrics.accuracy_score(y_test,y_pred)
precision =metrics.precision_score(y_test, y_pred,average='micro')
recall =  metrics.recall_score(y_test, y_pred,average='micro')

sns.heatmap(cm, annot=True, cbar=None, cmap="Blues")
plt.title("Matriz de confusión para el modelo gausiano bayesiano ingenuo"), plt.tight_layout()
plt.ylabel("Categoría real"), plt.xlabel("Predicted de la categoría")
plt.show()

print('Accuracy del Test para el modelo Bayesiano ingenuo: ',accuracy)
print('Accuracy del train para el modelo Bayesiano ingenuo:', accuracy_Entre)
print ("Precision para el modelo Bayesiano ingenuo:", precision )
print ("Recall para el modelo Bayesiano ingenuo: ",recall)



