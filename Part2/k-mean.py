from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import confusion_matrix
import seaborn as sns;

n_components = 2
k = 10

#PreProcessing des données

X = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_X_28x28.npy')
Y = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_y.npy')


Xr= X.reshape(70000,784)/255.0 #On reshape les données

x_train,x_test,y_train,y_test=train_test_split(Xr,Y,train_size=0.1) #On split entre test et train

##Pour afficher la distorsion en fonction des clusters##

disto = list()
for k in range(1, 15):
    kmeans = MiniBatchKMeans(n_clusters=k) #On prend un minibatch pour accéléerer les 15 KMeans
    kmeans = kmeans.fit(x_train)
    disto.append(kmeans.inertia_)

plt.figure(figsize=(15, 6)) #On trace la distorsion en fonction des clusters,
plt.plot(range(1, 15), disto, marker="x")# ie Somme des distances au carré des échantillons à leur centre de cluster le plus proche
plt.title("Distorsion en fonction du nombre de clusters")
plt.xlabel("Nombre de clusters K")
plt.ylabel("Distorsion")
plt.show()

##Fin de l'affichage de la distorsion en fonction du nombre de clusters


#On fait la PCA
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(x_train)#Résultat de la PCA
pca_test = pca.fit_transform(x_test)

#On fait le KMeans(un "vrai" cette fois)
kmeans = KMeans(init ="k-means++",n_clusters = 10)
kmeans = kmeans.fit(pca_result)


##On affiche le clustering sur x_train avec des marges

#La taille du maillage. Diminuer le maillage augmente la qualité.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

#On affiche la marge de décision et on affecte une couleur à chacune
x_min, x_max = pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1
y_min, y_max = pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Renvoie les labels pour chaque point dans le maillage (on utilise le précedents KMeans.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# On affiiche le résultat dans un diagramme coloré
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(pca_result[:, 0], pca_result[:, 1], 'k.', markersize=2)
# On marque les centroïdes du KMeans avec une croix blanche
centroids = kmeans.cluster_centers_ #les centroides de départ
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)

plt.title('K-means clustering sur x_train (données réduites après PCA)\n'
          "Les centroides sont marqués d'une croix blanche")

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


##Fin du diagramme de marges

#On imprime finalement le score d'homogénéité
print("Le score d'homogénéité entre la prédiction et la vérité terrain est de {}".format(homogeneity_score(y_train, kmeans.labels_)))

#On affiche les résultats sur le jeu de test
kmeans = KMeans(init ="k-means++",n_clusters = 10)
labels = kmeans.fit(pca_test).predict(pca_test)
plt.scatter(pca_test[:, 0], pca_test[:, 1], c=labels, s=10, cmap='viridis');
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=20, cmap='viridis', marker='x');
plt.title("Un clustering sur x_test. Les centroïdes sont marqués en noir")
plt.show()

#Test du K-Means sans PCA
kmeans = KMeans(init ="k-means++",n_clusters = 10)
kmeans = kmeans.fit(x_train)
print("Le score d'homogénéité entre la prédiction et la vérité terrain sans PCA est de {}".format(homogeneity_score(y_train, kmeans.labels_)))

#Matrice de confusion
labels_kmeans = kmeans.predict(x_test)
mat_kmeans = confusion_matrix(y_test, labels_kmeans)
sns.heatmap(mat_kmeans.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels="0123456789",
            yticklabels="0123456789")
plt.xlabel('Vérité terrain')
plt.ylabel('Label prédit');
plt.show()