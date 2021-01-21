from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans

r_state = 0
n_components = 2
k = 10
nb_labels = 10

#PreProcessing des données

X = np.load('/Part2/MNIST_X_28x28.npy')
Y = np.load('/Part2/MNIST_y.npy')


X= X.reshape(70000,784)/255.0 #On reshape pour correspondre à un vecteur pandas, plus maniable

print(X.shape)

print ("The shape of X is " + str(X.shape))
print ("The shape of Y is " + str(Y.shape))

x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.1) #On split entre test et train

print(x_train)

#Visualise an image
n= 2 #Enter Index here to View the image
#plt.imshow(X[n].reshape(x_train.shape[1], x_train.shape[2]), cmap = plt.cm.binary)
plt.show()
Y[n]

#PCA

pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(x_train)#Résultat de la PCA

#KMEANS

kmeans = KMeans(init ="k-means++",n_clusters = 10)
kmeans = kmeans.fit(reduced_data)


##Partie pour afficher le diagramme barre##

kmeans_labels = kmeans.labels_  # List of labels of each dataset
print("The list of labels of the clusters are " + str(np.unique(kmeans_labels)))

G = len(np.unique(kmeans_labels))  # Number of labels

# 2D matrix  for an array of indexes of the given label
cluster_index = [[] for i in range(G)]
for i, label in enumerate(kmeans_labels, 0):
    for n in range(G):
        if label == n:
            cluster_index[n].append(i)
        else:
            continue

print(x_train)
# Visualisation for clusters = clust
plt.figure(figsize=(20, 20));
clust = 8  # enter label number to visualise
num = 100  # num of data to visualize from the cluster
for i in range(1, num):
    plt.subplot(10, 10, i);  # (Number of rows, Number of column per row, item number)
    plt.imshow(X[cluster_index[clust][i + 500]].reshape(x_train.shape[1], x_train.shape[2]), cmap=plt.cm.binary);

plt.show()