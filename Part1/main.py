import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
X = np.load('../MNIST_X_28x28.npy')
Y = np.load('../MNIST_y.npy')
for i in range (100) :
    nb_sample = i
    plt.imshow(X[nb_sample])
    img_title = 'Classe' + str(Y[nb_sample])
    plt.title(img_title)
    plt.show()
    plt.clf


