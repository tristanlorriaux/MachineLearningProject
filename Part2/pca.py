from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd

n_components = 2 #pour la PCA

#PreProcessing des données

X = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_X_28x28.npy')
Y = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_y.npy')


Xr= X.reshape(70000,784)/255.0 #On reshape les données

print ("The shape of X is " + str(Xr.shape))
print ("The shape of Y is " + str(Y.shape))

x_train,x_test,y_train,y_test=train_test_split(Xr,Y,train_size=0.1) #On split entre test et train


#PCA

pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(x_train) #Résultat de la PCA

#On crée un tableau pandas pour l'affichage de la PCA sur Seaborn

feat_cols = [ 'pixel'+str(i) for i in range(x_train.shape[1]) ]
df = pd.DataFrame(x_train,columns=feat_cols)
df['Y'] = y_train
df['label'] = df['Y'].apply(lambda i: str(i))
print('Taille du tableau Pandas: {}'.format(df.shape))

df['pca-one'] = reduced_data[:,0]
df['pca-two'] = reduced_data[:,1]

#Affichage des résultas (variance explicite par composantes)
explained_variance = pca.explained_variance_ratio_.sum()
print('Variance explicite par composante principale: {}'.format(pca.explained_variance_ratio_))
print ("Avec {} composantes, l'explained variance is {}" .format(n_components, explained_variance))

#On trace sur un graphe 2D (axes = principales composantes) > on regarde la distrib par chiffre
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="Y",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3,
)
plt.title("PCA sur x_train",fontsize=25,
          color="red")
plt.show()
