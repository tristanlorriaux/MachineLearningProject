import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Flatten
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

#PreProcessing des données

DATA_PATH = '' #Insérez le chemin des fichiers 
X = np.load(DATA_PATH + '/MNIST_X_28x28.npy')
Y = np.load(DATA_PATH + '/MNIST_y.npy')

Xr= X.reshape(70000,784)/255.0 #On reshape les données
x_train,x_test,y_train,y_test=train_test_split(Xr,Y,train_size=0.7, random_state=41) #On split entre test et train

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))#Entrée adaptée au CNN, du 3x3x1
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Parameters
number_features=784
batch_size = 128 # Il s'agit de la taille de l'échantillon des entrées à traiter à chaque étape du trainig.
hidden_units = 256
dropout = 0.45
epochs = 10

# On construit le modèle
model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

#Activation

model.add(Dense(10, activation='softmax')) #La couche de sortie comporte 10 unités, suivies d'une fonction d'activation softmax. Les 10 unités correspondent aux 10 étiquettes, classes ou catégories possibles.
model.summary()
plot_model(model, "CNN_model.png", show_shapes=True)

#Opti
model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

fit = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_split=0.2)
scores = model.evaluate(x_test,y_test,verbose=2)
print("Perte du test:",scores[0])
print("Précision du test:",scores[1])

epochs_tab = np.arange(epochs)
plt.plot(epochs_tab,fit.history['val_accuracy'],label = 'test')
plt.plot(epochs_tab,fit.history['accuracy'],label = 'train')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Précision du jeu de données test et train')
plt.legend()
plt.show()

plt.plot(epochs_tab, fit.history['val_loss'], label = 'test')
plt.plot(epochs_tab, fit.history['loss'], label = 'train')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Perte du jeu de données test et train')
plt.legend()
plt.show()
