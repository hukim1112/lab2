from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


x = np.random.uniform(-1, 1, [1, 1000])[0]
y = np.sin(x*np.pi) + np.random.normal(0, 0.2, [1000])
data = [[i, j] for i, j in zip(x, y)]
X = np.array(data)
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  
print(pca.components_)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(x, y, s = 10, c ='b', marker="s", label='first')
ax1.scatter(pca.components_[:, 0], pca.components_[:, 1], s=10, c='r', marker="o", label='second')


ax1.arrow(0, 0, pca.components_[0, 0], pca.components_[0, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')

ax1.arrow(0, 0, pca.components_[1, 0], pca.components_[1, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')

plt.legend(loc='upper left');
plt.show()











