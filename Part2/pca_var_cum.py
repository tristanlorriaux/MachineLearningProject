from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd

n_components = 100 #pour la PCA

#PreProcessing des données

X = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_X_28x28.npy')
Y = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_y.npy')


Xr= X.reshape(70000,784)/255.0 #On reshape les données

print ("The shape of X is " + str(Xr.shape))
print ("The shape of Y is " + str(Y.shape))

x_train,x_test,y_train,y_test=train_test_split(Xr,Y,train_size=0.1) #On split entre test et train

#Explorons les variances cumulées en fonction de la dimension d'arrivée

tab_var_rat=[]
tab_compo=np.arange(0,n_components,1)

for i in tab_compo:

    pca = PCA(n_components=i)
    reduced_data = pca.fit_transform(x_train) #Résultat de la PCA

    explained_variance = pca.explained_variance_ratio_.sum()
    tab_var_rat.append(explained_variance)
    #print('Variance explicite par composante principale: {}'.format(pca.explained_variance_ratio_))
    print ("Avec {} composantes, l'explained variance is {}" .format(i, explained_variance))


plt.plot(tab_compo,tab_var_rat)
plt.xlabel('N_components')
plt.ylabel('Variance cumulée')
plt.title("Variances cumulées en fonction de la dimension d'arrivée de la PCA",fontsize='10')
plt.legend()
plt.show()