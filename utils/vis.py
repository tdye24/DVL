import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def vis(X, y, name):
    # 使用PCA进行降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    # 绘制PCA降维后的数据
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.title('PCA')

    # 绘制t-SNE降维后的数据
    plt.subplot(1, 2, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    plt.title('t-SNE')
    plt.savefig(f"{name}")
    plt.show()
