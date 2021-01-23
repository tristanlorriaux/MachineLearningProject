import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'C:/Users/titil/Desktop/MachineLearningProject/Part2' #Insérez le chemin des fichiers 
X = np.load(DATA_PATH + '/MNIST_X_28x28.npy')
Y = np.load(DATA_PATH + '/MNIST_y.npy')
for i in range (100) :
    nb_sample = i
    plt.imshow(X[nb_sample])
    img_title = 'Classe' + str(Y[nb_sample])
    plt.title(img_title)
    plt.show()
    plt.clf

moy = np.average(Y)
var = np.std(Y)
print("Moyenne : ", moy)
print("Variance : ", var)
