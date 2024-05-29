import streamlit as st
import json
import os
from embeddings import generate_embeddings, load_texts
from analysis import calculate_cosine_similarity, plot_similarity_heatmap, hierarchical_clustering, kmeans_clustering, pca_visualization, tsne_visualization

# Paths
data_dir = os.path.join(os.path.dirname(__file__), '../data')
text_file_path = os.path.join(data_dir, 'ai-texts.json')
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

def main():
    st.title("Content Embedding Analysis")

    # Load texts
    texts_with_ids = load_texts(text_file_path)

    # Generate embeddings if not already generated
    if not os.path.exists(embeddings_file_path):
        st.write("Generating embeddings for the texts...")
        embeddings = generate_embeddings([entry['text'] for entry in texts_with_ids])
        embeddings_data = [{'id': entry['id'], 'embedding': embedding} for entry, embedding in zip(texts_with_ids, embeddings)]
        with open(embeddings_file_path, 'w') as file:
            json.dump(embeddings_data, file, indent=2)
        st.write("Embeddings generated and saved.")
    else:
        embeddings_data = load_embeddings(embeddings_file_path)
        embeddings = [entry['embedding'] for entry in embeddings_data]
        section_labels = [entry['id'] for entry in embeddings_data]

    # Sidebar for selecting analysis type
    analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                         ("Cosine Similarity Heatmap", 
                                          "Hierarchical Clustering", 
                                          "K-means Clustering",
                                          "PCA Visualization",
                                          "t-SNE Visualization"))

    # Perform and display the selected analysis
    st.write(f"## {analysis_type}")
    if analysis_type == "Cosine Similarity Heatmap":
        similarity_matrix = calculate_cosine_similarity(embeddings)
        plot_similarity_heatmap(similarity_matrix, section_labels)
    elif analysis_type == "Hierarchical Clustering":
        hierarchical_clustering(embeddings, section_labels)
    elif analysis_type == "K-means Clustering":
        num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        kmeans_clustering(embeddings, section_labels, num_clusters=num_clusters)
    elif analysis_type == "PCA Visualization":
        pca_visualization(embeddings, section_labels)
    elif analysis_type == "t-SNE Visualization":
        tsne_visualization(embeddings, section_labels)

if __name__ == "__main__":
    main()
