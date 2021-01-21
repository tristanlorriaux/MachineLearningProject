import numpy as np
import matplotlib.pyplot as plt

X = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_X_28x28.npy')
Y = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_y.npy')

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
