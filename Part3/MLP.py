import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy


DATA_PATH = '' #Insérez le chemin des fichiers 
X = np.load(DATA_PATH + '/MNIST_X_28x28.npy')
Y = np.load(DATA_PATH + '/MNIST_y.npy')


Xr= X.reshape(70000,784)/255.0 #On reshape les données


x_train,x_test,y_train,y_test=train_test_split(Xr,Y,train_size=0.7, random_state=41) #On split entre test et train

#MLP donc tenseur 1D
num_labels = len(np.unique(y_train))

# Parameters
number_features=784
batch_size = 128 # Il s'agit de la taille de l'échantillon des entrées à traiter à chaque étape du trainig.
hidden_units = 256
dropout = 0.45
epochs = 20

# On construit le modèle (avec ReLu)
model = Sequential()

model.add(Dense(hidden_units, input_dim=number_features))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(num_labels))

#Activation
model.add(Activation('softmax')) #La couche de sortie comporte 10 unités, suivies d'une fonction d'activation softmax. Les 10 unités correspondent aux 10 étiquettes, classes ou catégories possibles.

plot_model(model, "MLP_model.png", show_shapes=True)
model.summary()

#Opti

# L'objectif de l'optimisation est de minimiser la fonction de perte (loss).
# L'idée est que si la perte est réduite à un niveau acceptable, le modèle a indirectement appris la fonction qui fait correspondre les entrées aux sorties.
# Les mesures de performance sont utilisées pour déterminer si notre modèle a appris.

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

#Une fonction de perte (loss) classique est une fonction de perte d'entropie croisée : la rétropropagation est simplement la somme des gradients dans le temps
#La précision est un bon critère de mesure pour les tâches de classification.
#Adam est un algorithme d'optimisation qui peut être utilisé à la place de la procédure classique de descente stochastique par gradient

fit = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
scores = model.evaluate(x_test,y_test, verbose=2)
print("Perte du test:",scores[0])
print("Précision du test:", scores[1])


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
