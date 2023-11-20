from __future__ import print_function

import copy
import time

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0 # shape: (70000, 784)
y = mnist.target

feat_cols = X.columns #[ 'pixel'+str(i) for i in range(X.shape[1]) ]

df=copy.deepcopy(X)
#df = pd.DataFrame(X,columns=feat_cols) # shape: (70000, 785)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

#### PCA on all samples
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values) #(70000, 784)-->(70000, 3)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#### PCA on first 10000 samples
N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1]
df_subset['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#### tSNE using raw data
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset) #(10000, 784)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]

#### tSNE using 50 PCA components
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(data_subset) # (10000, 784)-->(10000, 50)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50) # (10000, 50)-->(10000, 2)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]


