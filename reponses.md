# 1.	The datasets: MNIST

Load the dataset 

_WORK 1_ : Les données sont dans le tableau d’images X, aléatoirement. Les vérités terrains sont contenues dans le tableau Y.  

_WORK 2_ : De manière purement arbitraire : oui. 
NB : On pourrait faire la variance et la moyenne sur les vérités terrains pour observer.

# 2.  Unsupervised Machine Learning

**2.1. Dimensionality reduction**

We use the MNIST dataset.

Pictures from MNIST have 784 pixels (28x28 grayscale picture), i.e. 𝑥(𝑖)∈𝑋 ⊂ ℝ784 we are in a (relatively) high-dimensional space. It is difficult to “see” how the data are distributed and if some intrinsic characteristics between the features exist, that could help further analysis / tasks.


WORK: Perform a Principal Component Analysis (PCA) with sklearn.
Réalisé


WORK: Try different n_components.

Faire la boucle 


WORK: Try to display some MNIST pictures with different n_components.
An interesting feature is PCA.explained_variance_ratio_.
Pas encore réussi :/


WORK: Explain these values according to your understanding of PCA and use these values to fit n_components.
