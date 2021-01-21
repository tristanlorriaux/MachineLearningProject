from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.cluster import homogeneity_score


r_state = 0
n_components = 2
k = 10
nb_labels = 10

X = np.load('/Part2/MNIST_X_28x28.npy')
Y = np.load('/Part2/MNIST_y.npy')

#On reshape pour correspondre à un vecteur pandas, plus maniable
X= X.reshape(70000,784)/255.0

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['Y'] = Y
df['label'] = df['Y'].apply(lambda i: str(i))

print('Size of the dataframe: {}'.format(df.shape))

train_x = pd.DataFrame(X,columns=feat_cols)
train_y = pd.DataFrame(Y,columns=['Y'])

X, Y = None, None

x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,train_size=0.1)
#print (x_train,y_train)

n_digits = len(np.unique(train_y))
print(n_digits)

#kmeans = KMeans(n_clusters=n_digits,random_state=r_state)
#kmeans_result = kmeans.fit(x_train)

#disto = list()
#for k in range(1, 15):
#    kmeans = MiniBatchKMeans(n_clusters=k)
#    kmeans = kmeans.fit(x_train)
#    disto.append(kmeans.inertia_)

#plt.figure(figsize=(15, 6)) #On trace la distorsion en fonction des clusters,
#plt.plot(range(1, 15), disto, marker="o")# ie Somme des distances au carré des échantillons à leur centre de cluster le plus proche
#plt.show()

#On fait la PCA

pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(x_train)#Résultat de la PCA

kmeans = KMeans(init ="k-means++",n_clusters = 10)

kmeans = kmeans.fit(pca_result)


# La taille du maillage. Diminuer le maillage pour augmenter la qualité de la VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# On affiche la marge de décision et on affecte une couleur à chacune
x_min, x_max = pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1
y_min, y_max = pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Afficher le résultat dans un diagramme coloré
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(pca_result[:, 0], pca_result[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_ #les centroides de départ
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)

plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

print(homogeneity_score(y_train['Y'], kmeans.labels_))