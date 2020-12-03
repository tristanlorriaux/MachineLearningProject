# 1.	The datasets: MNIST

Load the dataset 

_WORK 1_ : Les données sont dans le tableau d’images X, aléatoirement. Les vérités terrains sont contenues dans le tableau Y.  

_WORK 2_ : De manière purement arbitraire : oui. 
NB : On pourrait faire la variance et la moyenne sur les vérités terrains pour observer.

# 2.  Unsupervised Machine Learning

**2.1. Dimensionality reduction**

_We use the MNIST dataset._

_Pictures from MNIST have 784 pixels (28x28 grayscale picture), i.e. 𝑥(𝑖)∈𝑋 ⊂ ℝ784 we are in a (relatively) high-dimensional space. It is difficult to “see” how the data are distributed and if some intrinsic characteristics between the features exist, that could help further analysis / tasks._


WORK: Perform a Principal Component Analysis (PCA) with sklearn.
Réalisé


WORK: Try different n_components.

A essayer sur le code (le graphe ne fonctionne que en 2 compo, logique pour du 3 compo il faut du 3D :/)


WORK: Try to display some MNIST pictures with different n_components.
An interesting feature is PCA.explained_variance_ratio_.
fait (voir graphes)


WORK: Explain these values according to your understanding of PCA and use these values to fit n_components.

La variance expliquée fait référence à la variance expliquée par chacune des principales composantes (vecteurs propres).

**2.2. Data clustering**

_We use the MNIST dataset._

_Our goal is to cluster X. We will use classical approaches: K-MEANS and EM with Gaussian mixture
(see Tips below for documentation links)._


WORK: Split X (and y) in a train and test sets with the sklearn method: split_train_test.


WORK: With sklearn, perform K-MEANS. Play with the parameter K as well as the initialization


(KMEANS++, random, or fixed array).
WORK: For the correct K, evaluate how good is this partition (with the knowledge of y).


WORK: Using the PCA performed in section 2.1. apply K-MEANS with K=10 and n_components
= 2. Display the partition and comment.


WORK: Briefly explain what is the main difference between K-MEANS and EM with Gaussian
Mixture.


WORK: Do the same job with the EM-clustering using the good K parameter (10 for MNIST).
Comment your results.
