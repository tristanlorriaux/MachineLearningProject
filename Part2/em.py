from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns

n_components = 2
k = 10

#PreProcessing des données

X = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_X_28x28.npy')
Y = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_y.npy')


Xr= X.reshape(70000,784)/255.0 #On reshape les données

x_train,x_test,y_train,y_test=train_test_split(Xr,Y,train_size=0.1) #On split entre test et train


#On fait la PCA
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(x_train)#Résultat de la PCA
pca_test = pca.fit_transform(x_test)

#On applique un algo Expectation-Maximization

GM = GaussianMixture(n_components=k,covariance_type='full')
GM.fit(pca_result)

#On imprime finalement les nombres à consulter

print(GM.means_)
print("----------------------")
print(GM.covariances_)
print("La probabilité de mélange gaussien est de {}".format(math.exp(GM.score(pca_test)))) #On cherche à minimiser le log

#On affiche les résultats sur le jeu de test
GM = GaussianMixture(n_components=k,covariance_type='full')
labels = GM.fit(pca_test).predict(pca_test)
plt.scatter(pca_test[:, 0], pca_test[:, 1], c=labels, s=10, cmap='viridis');
plt.scatter(GM.means_[:, 0], GM.means_[:, 1], c='black', s=20, cmap='viridis', marker='x')
plt.title("Un autre clustering sur x_test. Les moyennes sont marqués en noir")
plt.show()

#Clustering sans PCA (/!\ LONG A CALCULER)
#GM = GaussianMixture(n_components=k,covariance_type='full')
#GM.fit(x_train)
#print("La probabilité de mélange gaussien est de {}".format(math.exp(GM.score(x_test))))

#Matrice de confusion
labels_em = GM.predict(pca_test)
mat_gm = confusion_matrix(y_test, labels_em)
sns.heatmap(mat_gm.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels="0123456789",
            yticklabels="0123456789")
plt.xlabel('Vérité terrain')
plt.ylabel('Label prédit');
plt.show()