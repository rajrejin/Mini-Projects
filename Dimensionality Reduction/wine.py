import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('D:/FAU/4. WS 23/DSS/Exercises/My-Projects/Dimensionality Reduction/winequality-red.csv')

# Separate features and target
features = df.drop('quality', axis=1)
target = df['quality']

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=2)
features_pca_standardized = pca.fit_transform(features_normalized)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
features_tsne_standardized = tsne.fit_transform(features_normalized)

# Apply UMAP
umap = umap.UMAP(n_components=2, random_state=42).fit_transform(features_normalized)

# Create PCA plot
plt.figure(figsize=(10, 10))
scatter = plt.scatter(features_pca_standardized[:, 0], features_pca_standardized[:, 1], c=target, cmap='viridis')
plt.title('PCA')
plt.colorbar(scatter, label='Quality')

# Create t-SNE plot
plt.figure(figsize=(10, 10))
scatter = plt.scatter(features_tsne_standardized[:, 0], features_tsne_standardized[:, 1], c=target, cmap='plasma')
plt.title('t-SNE')
plt.colorbar(scatter, label='Quality')

# Create UMAP plot
plt.figure(figsize=(10, 10))
scatter = plt.scatter(umap[:, 0], umap[:, 1], c=target, cmap='cividis')
plt.title('UMAP')
plt.colorbar(scatter, label='Quality')

plt.show()