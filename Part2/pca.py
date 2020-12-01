from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import plotter
import tikzplotlib

n_components = 2

X = np.load('../MNIST_X_28x28.npy')
Y = np.load('../MNIST_y.npy')

def image2vec(image):
    return image.flatten()


def vec2image(vec):
    return vec.reshape(28, 28)


n_samples = len(X)
print('On a 70000 images de 28X28')
print(X.shape)
print("------------")

x = np.array([image2vec(im) for im in X])
pca = PCA(n_components=n_components)
x_pca=pca.fit_transform(x)
X_rec = pca.inverse_transform(x_pca)
images_rec = [vec2image(x) for x in X_rec]

# print result
explained_variance = pca.explained_variance_ratio_.sum()
print('Using {} components, explained variance is {}'.format(n_components, explained_variance))



