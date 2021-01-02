import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

#PreProcessing des données

X = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_X_28x28.npy')
Y = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_y.npy')


Xr= X.reshape(70000,784)/255.0 #On reshape les données


x_train,x_test,y_train,y_test=train_test_split(Xr,Y,train_size=0.1) #On split entre test et train

#MLP donc tenseur 1D
num_labels = len(np.unique(y_train))

image_size = x_train.shape[1]
input_size = image_size * image_size
# Parameters
batch_size = 128 # Il s'agit de la taille de l'échantillon des entrées à traiter à chaque étape du trainig.
hidden_units = 256
dropout = 0.45

# On construit le modèle (avec ReLu)
model = Sequential()

model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(num_labels))

#Activation
model.add(Activation('softmax')) #La couche de sortie comporte 10 unités, suivies d'une fonction d'activation softmax. Les 10 unités correspondent aux 10 étiquettes, classes ou catégories possibles.
model.summary()

#Opti

# L'objectif de l'optimisation est de minimiser la fonction de perte (loss).
# L'idée est que si la perte est réduite à un niveau acceptable, le modèle a indirectement appris la fonction qui fait correspondre les entrées aux sorties.
# Les mesures de performance sont utilisées pour déterminer si notre modèle a appris.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Une fonction de perte (loss) classique est une fonction de perte d'entropie croisée : la rétropropagation est simplement la somme des gradients dans le temps
#La précision est un bon critère de mesure pour les tâches de classification.
#Adam est un algorithme d'optimisation qui peut être utilisé à la place de la procédure classique de descente stochastique par gradient
model.fit(x_train, y_train, epochs=20, batch_size=batch_size)