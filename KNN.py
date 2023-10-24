import pandas as pd
import seaborn as sb
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.width', 100)

egg_mass = 63.0

#Ucitavanje i prikaz podataka

data = pd.read_csv("cakes.csv")
data['eggs'] = data['eggs']*egg_mass

print(data.head(5))
print(data.info())
print(data.describe())


# NEMA NEDOSTAJUCIH PODATAKA

# Kodiranje izlaza cupcake -> 0 ; muffin -> 1
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

labels = data['type']
data = data.drop('type',axis = 1)

K0 = data.loc[labels == 0, :] # cupcakes
K1 = data.loc[labels == 1, :] # muffins



# Korelaciona matrica
plt.figure()
sb.heatmap(data.corr(), annot= True)
plt.show()


# 1D grafik - pripadanje klasi u zavisnosti od atributa
def decart(labels,feature):
    plt.figure()
    plt.scatter(K0[feature],np.zeros(len(K0[feature])))
    plt.scatter(K1[feature],np.zeros(len(K1[feature])))
    plt.legend(['Cupcake','Muffin'])
    plt.title(feature)
    plt.xlabel('[g]')
    plt.show()
    


for feature in data.columns:
    decart(labels,feature)

# 2D grafik
def decart2D(f1,f2):
    plt.figure()
    plt.scatter(K0[f1],K0[f2])
    plt.scatter(K1[f1],K1[f2])
    plt.legend(['Cupcake','Muffin'])
    plt.xlabel(f1 + "[g]")
    plt.ylabel(f2 + "[g]")
    plt.show()
    
#decart2D('sugar','butter')
    
# Izbor obelezja
data = data[['eggs','sugar','butter','baking_powder']]

# Normalizacija
scaler = MinMaxScaler()
scaler.fit(data)
data = pd.DataFrame(scaler.transform(data),columns = data.columns)



#Train/Val/Test split
X_train, X_test, Y_train, Y_test = train_test_split(data,labels,shuffle = True, random_state = 20,test_size = 0.2)
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,shuffle = True, random_state = 5, test_size = 0.2)

#Built-in KNN
acc_max = 0
k_opt = 0

# pronalazenje optimalnog k racunanjem tacnosti na validacionom skupu
for k in range(1,int(len(X_train))):
    KNN = KNeighborsClassifier(n_neighbors=k)
    KNN.fit(X_train,Y_train)

    Y_pred_built_in = KNN.predict(X_val)
    acc = accuracy_score(Y_val,Y_pred_built_in)*100
    if(acc > acc_max):
        acc_max = acc
        k_opt = k
    
# KNN sa optimalnim k
KNN = KNeighborsClassifier(n_neighbors=k_opt)
KNN.fit(X_train,Y_train) # treniranje
Y_pred_built_in = KNN.predict(X_test) # predikcija
acc = accuracy_score(Y_test,Y_pred_built_in)*100 # tacnost na test skupu
print("--------------")
print("BUILT-IN KNN")
print("--------------")
print("k optimalno: " + str(k_opt))
print("Tacnost na validacionom skupu: " + str(acc_max) + "%") 
print("Tacnost na test skupu: " + str(acc) + "%")
#print(KNN.get_params())
    

# Moj KNN

k = k_opt # koristimo k pronadjeno u prethodnom delu
Y_pred_mine = []

# racunanje Euklidove distance
def distance(test_point,train_point):
    dist = 0
    for i in range(len(test_point)):
        dist += (test_point[i] - train_point[i])**2

    return dist**0.5

# predikcija kojoj klasi pripada tacka na osnovu trening skupa
def predict(test_point,X_train):
    neighbours = []
    neighbours_class = []
    
    # racunanje distanci do svih odabiraka iz trening skupa
    for i in range(len(X_train)):
        dist = distance(test_point,X_train.iloc[i,:])
        
        # ubacivanje distance tako da zadrzimo nerastuci poredak
        j = 0            
        while(j < len(neighbours) and neighbours[j] < dist):
            j += 1
    
        neighbours.insert(j, dist)
        #pamtimo i klasu odabirka
        neighbours_class.insert(j,Y_train.iloc[i])
    
    
    #provera koja klasa je u vecini u prvih k komsija
    score = 0
    for i in range(k):
        if neighbours_class[i] == 0:
            score -= 1
        elif neighbours_class[i] == 1:
            score += 1

    if(score <= 0):
        pred = 0
    elif(score > 0):
        pred = 1
        
    return pred

# predikcije na test skupu
for i in range(len(X_test)):
    Y_pred_mine.append(predict(X_test.iloc[i,:],X_train))
    
# racunanje tacnosti na test skupu
acc = 0
for i in range(len(Y_test)):
    if(Y_test.iloc[i] == Y_pred_mine[i]):
        acc += 1

acc = acc/len(Y_test)*100
print("--------------")
print("Moj KNN")
print("--------------")
print("Tacnost na test skupu: " + str(acc) + "%")