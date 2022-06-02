#!/usr/bin/env python
# coding: utf-8

# # Proyecto de física experimental: Comparación de algoritmos de clasificación no supervisados aplicados al set de datos 17 de Sloan Digital Sky Survey (SDSS)
# 
# ## Universidad del Valle de Guatemala
# ### Javier Alejandro Mejía Alecio (20304)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from scipy.stats import shapiro
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz
import random
import scipy.stats as stats
import timeit
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression



Data = pd.read_csv('Skyserver_Data.csv')


print(Data.shape)


Data = Data.replace(['STAR','GALAXY','QSO'], [0,1,2])


# ### Prueba de Normalidad de las variables

# In[5]:


CN = Data.columns.values


# ### Se toma una muestra de 50,000 mediciones

# In[6]:


Muestra = Data.sample(50000, random_state = 123)


# In[7]:


Muestra = Muestra.reset_index()


# ### Eliminación de puntos atípicos

# In[8]:


clf = LocalOutlierFactor()
y_pred = clf.fit_predict(Muestra)


# In[9]:


x_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

#threshold
threshold = np.quantile(x_score , .10)                                            
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()


# In[10]:


print(outlier_index)


# In[11]:


Muestra.drop(outlier_index, inplace=True)


# In[12]:


correlation_mat = Muestra.corr()


# In[13]:


correlation_mat["class"].sort_values()


# Se toman las variables con una correlación mayor a |.1|

# In[14]:


Muestra = Muestra[['mjd', 'plate', 'specobjid', 'i', 'u','r', 'g', 'redshift', 'class']]


# In[15]:


sns.pairplot(Muestra, hue="class",  diag_kws={'bw': .15})
plt.tight_layout()


# In[16]:


vif_data = pd.DataFrame() 
G = Muestra
vif_data["feature"] = G.columns
  
vif_data["VIF"] = [variance_inflation_factor(G.values, i) 
                          for i in range(len(G.columns))] 
  
print(vif_data)


# In[18]:


Muestra = Muestra.drop(['plate', 'specobjid'], axis = 1)


# In[19]:


Muestra['class'] = Muestra['class'].astype('category')


# Debido al resultado del VIF se decide eliminar la variable plate para evitar problmea de múlticolinealidad

# ## Se inicia separando los datos de la muestra para tener un grupo de test y train

# In[20]:


y = Muestra.pop("class") #La variable respuesta
X = Muestra #El resto de los datos

random.seed(123)

# In[23]:


X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7, random_state = 123)
print(X_train)


# In[26]:



# ### árbol de decisión

# In[27]:


arbol = DecisionTreeClassifier(max_depth=4, random_state=1) 
arbol = arbol.fit(X_train, y_train) 

tree.plot_tree(arbol,feature_names= Muestra.columns, class_names=['0','1','2'],filled=True )

y_predf = arbol.predict(X_train)


y_pred = arbol.predict(X_test)
print ("Accuracy entrenamiento:",metrics.accuracy_score(y_train, y_predf))
print ("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ("Precision:", metrics.precision_score(y_test,y_pred,average='weighted') )
print ("Recall: ", metrics.recall_score(y_test,y_pred,average='weighted'))

matrix = confusion_matrix(y_test, y_pred)

print(matrix)

# Random Forest:

RF = RandomForestClassifier(n_estimators=100,max_depth = 4, random_state = 1)
RF.fit(X_train, y_train)
plt.figure()
_ = tree.plot_tree(RF.estimators_[49], feature_names=X.columns, filled=True)
plt.title('Random Forest')
plt.show()

y_predf = RF.predict(X_train)


y_pred = RF.predict(X_test)
print ("Accuracy entrenamiento:",metrics.accuracy_score(y_train, y_predf))
print ("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print ("Precision:", metrics.precision_score(y_test,y_pred,average='weighted') )
print ("Recall: ", metrics.recall_score(y_test,y_pred,average='weighted'))
matrix = confusion_matrix(y_test, y_pred)

print(matrix)

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

print('Confusion matrix for Naive Bayes\n',cm)
print('Accuracy del Test: ',accuracy)
print('Accuracy del train:', accuracy_Entre)




