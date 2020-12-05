from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans

r_state = 0

X = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_X_28x28.npy')
Y = np.load('C:/Users/titil/Desktop/MachineLearningProject/Part2/MNIST_y.npy')

#On reshape pour correspondre à un vecteur pandas, plus maniable
X= X.reshape(70000,784)/255.0

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['Y'] = Y
df['label'] = df['Y'].apply(lambda i: str(i))

print('Size of the dataframe: {}'.format(df.shape))

train_x = pd.DataFrame(X,columns=feat_cols)
train_y = pd.DataFrame(Y,columns=['Y'])

X, Y = None, None

x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.25)
#print (x_train,y_train)

n_digits = len(np.unique(train_y))
print(n_digits)

#kmeans = KMeans(n_clusters=n_digits,random_state=r_state)
#kmeans_result = kmeans.fit(x_train)

disto = list()
for k in range(1, 15):
    kmeans = MiniBatchKMeans(n_clusters=k)
    kmeans = kmeans.fit(x_train)
    disto.append(kmeans.inertia_)


plt.figure(figsize=(15, 6)) #On trace la distorsion en fonction des clusters,
plt.plot(range(1, 15), disto, marker="o")# ie Somme des distances au carré des échantillons à leur centre de cluster le plus proche
plt.show()