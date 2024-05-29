import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import streamlit as st

# Paths
data_dir = os.path.join(os.path.dirname(__file__), '../data')
embeddings_file_path = os.path.join(data_dir, 'embeddings.json')

def load_embeddings(file_path):
    """
    Load embeddings from the specified JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing embeddings.
        
    Returns:
        list: A list of dictionaries with 'id' and 'embedding'.
    """
    with open(file_path, 'r') as file:
        embeddings_data = json.load(file)
    return embeddings_data

def calculate_cosine_similarity(embeddings):
    """
    Calculate the cosine similarity matrix for a given list of embeddings.
    
    Args:
        embeddings (list): List of embeddings for the sections.
        
    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    num_sections = len(embeddings)
    similarity_matrix = np.zeros((num_sections, num_sections))

    for i in range(num_sections):
        for j in range(num_sections):
            if i != j:
                similarity_matrix[i][j] = 1 - cosine(embeddings[i], embeddings[j])
            else:
                similarity_matrix[i][j] = 1.0  # Similarity with itself

    return similarity_matrix

def plot_similarity_heatmap(similarity_matrix, section_labels):
    """
    Generate and display a heatmap for the cosine similarity matrix.
    
    Args:
        similarity_matrix (np.ndarray): Cosine similarity matrix.
        section_labels (list of str): Labels for each section to be used in the heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, xticklabels=section_labels, yticklabels=section_labels, annot=True, cmap="viridis")
    plt.title("Cosine Similarity Heatmap")
    st.pyplot(plt.gcf())
    plt.close()

def hierarchical_clustering(embeddings, section_labels):
    """
    Perform hierarchical clustering and plot a dendrogram.
    
    Args:
        embeddings (list): List of embeddings for the sections.
        section_labels (list of str): Labels for each section to be used in the dendrogram.
    """
    linked = linkage(embeddings, method='ward')
    
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=section_labels, distance_sort='descending')
    plt.title("Hierarchical Clustering Dendrogram")
    st.pyplot(plt.gcf())
    plt.close()

def kmeans_clustering(embeddings, section_labels, num_clusters=3):
    """
    Perform K-means clustering and plot the clusters.
    
    Args:
        embeddings (list): List of embeddings for the sections.
        section_labels (list of str): Labels for each section.
        num_clusters (int): Number of clusters to form.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 7))
    for i in range(num_clusters):
        points = principal_components[labels == i]
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i+1}')
    for i, label in enumerate(section_labels):
        plt.annotate(label, (principal_components[i, 0], principal_components[i, 1]))
    plt.legend()
    plt.title("K-means Clustering")
    st.pyplot(plt.gcf())
    plt.close()

def pca_visualization(embeddings, section_labels):
    """
    Perform PCA to reduce dimensionality and plot the result.
    
    Args:
        embeddings (list): List of embeddings for the sections.
        section_labels (list of str): Labels for each section.
    """
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(embeddings)
    
    fig = px.scatter(x=principal_components[:, 0], y=principal_components[:, 1], text=section_labels)
    fig.update_traces(textposition='top center')
    fig.update_layout(title="PCA Dimensionality Reduction", xaxis_title="PCA Component 1", yaxis_title="PCA Component 2")
    st.plotly_chart(fig)

def tsne_visualization(embeddings, section_labels):
    """
    Perform t-SNE to reduce dimensionality and plot the result.
    
    Args:
        embeddings (list): List of embeddings for the sections.
        section_labels (list of str): Labels for each section.
    """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_components = tsne.fit_transform(embeddings)
    
    fig = px.scatter(x=tsne_components[:, 0], y=tsne_components[:, 1], text=section_labels)
    fig.update_traces(textposition='top center')
    fig.update_layout(title="t-SNE Dimensionality Reduction", xaxis_title="t-SNE Component 1", yaxis_title="t-SNE Component 2")
    st.plotly_chart(fig)

def main():
    # Load embeddings from the JSON file
    embeddings_data = load_embeddings(embeddings_file_path)
    embeddings = [entry['embedding'] for entry in embeddings_data]
    section_labels = [entry['id'] for entry in embeddings_data]
    
    # Perform analyses
    similarity_matrix = calculate_cosine_similarity(embeddings)
    plot_similarity_heatmap(similarity_matrix, section_labels)
    hierarchical_clustering(embeddings, section_labels)
    kmeans_clustering(embeddings, section_labels, num_clusters=3)
    pca_visualization(embeddings, section_labels)
    tsne_visualization(embeddings, section_labels)

if __name__ == "__main__":
    main()
