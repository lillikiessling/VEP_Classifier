from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import FastICA
import umap
import numpy as np
import matplotlib.pyplot as plt

class DimensionalityReducer:
    def __init__(self, n_pca=6, n_lda=2, n_ica=6, random_state=42):
        self.n_pca = n_pca
        self.n_lda = n_lda
        self.n_ica = n_ica
        self.random_state = random_state


    def pca(self, X_train, X_test):
        n_comp = min(self.n_pca, X_train.shape[1] - 1)
        if n_comp < 1:
            return X_train, X_test  # skip reduction if too few features
        pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca
    
    def ica(self, X_train, X_test):
        n_comp = min(self.n_ica, X_train.shape[1] - 1)
        if n_comp < 1:
            return X_train, X_test
        ica = FastICA(n_components=n_comp, random_state=self.random_state)
        X_train_ica = ica.fit_transform(X_train)
        X_test_ica = ica.transform(X_test)
        return X_train_ica, X_test_ica

    
    def kernel_pca(self, X_train, X_test, kernel='rbf'):
        n_comp = min(self.n_pca, X_train.shape[1] - 1)
        if n_comp < 1:
            return X_train, X_test
        kpca = KernelPCA(n_components=n_comp, kernel=kernel, random_state=self.random_state)
        X_train_kpca = kpca.fit_transform(X_train)
        X_test_kpca = kpca.transform(X_test)
        return X_train_kpca, X_test_kpca


    def lda(self, X_train, y_train, X_test):
        lda = LinearDiscriminantAnalysis(n_components=self.n_lda)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)
        return X_train_lda, X_test_lda
    
    
