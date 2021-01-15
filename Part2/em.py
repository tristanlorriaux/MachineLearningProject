from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import homogeneity_score

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

#On applique un algo Expectation-Maximization

my_em = GaussianMixture(n_components=k,covariance_type='full')
my_em.fit(pca_result)


##On affiche le clustering sur x_train avec des marges

#La taille du maillage. Diminuer le maillage augmente la qualité.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

#On affiche la marge de décision et on affecte une couleur à chacune
x_min, x_max = pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1
y_min, y_max = pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Renvoie les labels pour chaque point dans le maillage (on utilise le précedents KMeans.
Z = my_em.predict(np.c_[xx.ravel(), yy.ravel()])

# On affiiche le résultat dans un diagramme coloré
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(pca_result[:, 0], pca_result[:, 1], 'k.', markersize=2)


plt.title('Gaussian Mixture sur x_train (données réduites après PCA)\n')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


##Fin du diagramme de marges

#On imprime finalement le score d'homogénéité
print("Le score d'homogénéité entre la prédiction et la vérité terrain est de {}".format(homogeneity_score(y_train, my_em.predict(pca_result))))