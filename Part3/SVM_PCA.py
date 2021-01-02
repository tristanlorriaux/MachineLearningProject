import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.decomposition import PCA

n_components = 2

# PreProcessing des données

X = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_X_28x28.npy')
Y = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_y.npy')


Xr= X.reshape(70000,784)/255.0 # On reshape les données


#PCA

pca = PCA(n_components=n_components)
reduced_data = pca.fit_transform(Xr) #Résultat de la PCA

x_train,x_test,y_train,y_test=train_test_split(reduced_data,Y,train_size=0.5,shuffle=False) #On split entre test et train
print("Préprocessing terminé")


#SVM

clf = svm.SVC(gamma=0.001)

# Apprentissage sur les données train
clf.fit(x_train, y_train)
print("Apprentissage terminé")
# Prédiction des labels sur les données test
predicted = clf.predict(x_test)
print("Prédiction terminée")


# Affichage de 4 images avec leur prédiction au dessus du jeu de données test
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X, predicted):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')


# Rapport des métriques pour évaluer la classification

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")


# Matrice de confusion

disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()