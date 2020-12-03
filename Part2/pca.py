from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

n_components = 2 #pour la PCA

#On importe les bdd numpy
X = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_X_28x28.npy')
Y = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_y.npy')

#On reshape pour correspondre à un vecteur pandas, plus maniable
X= X.reshape(70000,784)/255.0

#On vérifie le reshape
print(X.shape, Y.shape)

#On convertit la matrice et le vecteur dans un tableau Pandas
feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['Y'] = Y
df['label'] = df['Y'].apply(lambda i: str(i))
X, Y = None, None
print('Size of the dataframe: {}'.format(df.shape))

np.random.seed(68)#On crée un sous-ensemble aléatoire des 70000 chiffres : on crée une permutation aléatoire des nombres 0 à 69999
rndperm = np.random.permutation(df.shape[0])

#On affiche des figures de 16 nombres
plt.gray()
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )#On va chercher le label du numéro dans le tableau pandas
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))#On va chercher les chiffres
plt.show()

pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(df[feat_cols].values)#Résultat de la PCA
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]

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
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
plt.show()