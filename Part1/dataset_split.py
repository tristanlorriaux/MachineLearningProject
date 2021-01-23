import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = 'C:/Users/Gabin Durteste/Downloads/MNIST' #Insérez le chemin des fichiers 
X = np.load(DATA_PATH + '/MNIST_X_28x28.npy')
Y = np.load(DATA_PATH + '/MNIST_y.npy')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print(len(X_train))
nb_sample = 1
plt.imshow(X_train[nb_sample])
img_title = 'Classe' + str(Y[nb_sample])
plt.title(img_title)
plt.show()
plt.clf

print(Y_train[nb_sample])
