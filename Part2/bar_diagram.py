from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

n_components = 2
k = 10


#PreProcessing des données

DATA_PATH = 'C:/Users/Gabin Durteste/Downloads/MNIST' #Insérez le chemin des fichiers 
X = np.load(DATA_PATH + '/MNIST_X_28x28.npy')
Y = np.load(DATA_PATH + '/MNIST_y.npy')


Xr= X.reshape(70000,784)/255.0 #On reshape les données

x_train,x_test,y_train,y_test=train_test_split(Xr,Y,train_size=0.1) #On split entre test et train


#PCA

pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(x_train) #Résultat de la PCA

#KMEANS

kmeans = KMeans(init ="k-means++",n_clusters = 10)
kmeans = kmeans.fit(reduced_data) #Résultat du kmeans



##Partie sur l'affichage des clusters##

kmeans_labels = kmeans.labels_  # Liste de labels du jeu de données
print("La liste des labels de clusters : " + str(np.unique(kmeans_labels)))

G = len(np.unique(kmeans_labels))  # Nombre de labels


cluster_index = [[] for i in range(G)] # Matrice deux dim. pour un ensemble d'indexes pour un label donné
for i, label in enumerate(kmeans_labels, 0):
    for n in range(G):
        if label == n:
            cluster_index[n].append(i)
        else:
            continue

# Graphe pour un cluster = clust (à rentrer à la main)

plt.figure(figsize=(20, 20));

clust = 1  # Entrer clust ici
num = 100  # Nombre de données du cluster à afficher

for i in range(1, num):
    plt.subplot(10, 10, i);  # (Nombre de lignes, de colomnes par lignes, num de l'élément)
    plt.imshow(x_train[cluster_index[clust][i + 500]].reshape(X.shape[1], X.shape[2]), cmap=plt.cm.binary);
plt.show()


##Fin de la partie pour l'affichage d'un extrait de cluster

##Partie pour le diagramme en barres

Y_clust = [[] for i in range(G)]
for n in range(G):
    Y_clust[n] = y_train[cluster_index[n]] #Y_clust[0] contient un ensemble de labels corrects de y_train pour cluster_index[0]
    assert(len(Y_clust[n]) == len(cluster_index[n])) #Vérification dimensionnelle(opt)

#Compteur de chaque label dans un cluster donné
def counter(cluster):
    unique, counts = np.unique(cluster, return_counts=True)
    label_index = dict(zip(unique, counts))
    return label_index

label_count= [[] for i in range(G)]
for n in range(G):
    label_count[n] = counter(Y_clust[n])

label_count[1] #Nombre d'élements d'un label donné dans le cluster 1

class_names = [0,1,2,3,4,5,6,7,8,9] #dico des noms de labels

def plotter(label_dict): #Fonction pour afficher un diagramme barres : nombre d'items par label dans un cluster donné
    plt.bar(range(len(label_dict)), list(label_dict.values()), align='center')
    a = []
    for i in [*label_dict]: a.append(class_names[i])
    plt.xticks(range(len(label_dict)), list(a), rotation=45, rotation_mode='anchor')


plt.figure(figsize=(20,20))# On boucle la fonction !
for i in range (1,11):
    plt.subplot(5, 2, i)
    plotter(label_count[i-1])
    plt.title("Cluster" + str(i-1))
plt.show()

##Fin de la partie pour l'affichage d'un diagramme barres
