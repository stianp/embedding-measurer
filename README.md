### README.md

# Content Embedding Analysis

## Project Overview

This project is designed to analyze the content of various sections of a website by using embeddings to assess how closely the content matches the overall content on the site. By leveraging natural language processing (NLP) techniques, specifically embeddings, we can visualize and understand the distribution and similarity of content within a website. This project provides tools for visualizing these embeddings and identifying how closely related different sections of content are to each other.

## Why We're Doing This

Google has been using embeddings to understand and evaluate website content for a long time. As NLP technology has advanced, so too have the techniques for understanding content. As SEOs, it's crucial to understand this technology, particularly when engaging in media partnerships or assessing content relevance. This understanding can help in various ways:

- **Content Relevance**: By visualizing content embeddings, we can ensure that the content aligns with the overall theme of the site, avoiding potential penalties from Google.
- **Media Partnerships**: Before entering partnerships, we can assess how closely related the partner's content is to our own, ensuring a good match.
- **Parasite SEO and Backlink Building**: Understanding the embedding structure of a site helps in assessing the relevance and quality of backlinks or third-party content, thereby avoiding penalties.
- **Outlier Detection**: Identify content that is significantly different from the rest of the site, which could indicate irrelevant or problematic content.

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/content-embedding-analysis.git
   cd content-embedding-analysis/src
   ```

2. **Install required packages**:
   ```sh
   pip install -r ../requirements.txt
   ```

3. **Set up your OpenAI API key**:
   Create a `.env` file in the root directory of the project and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   ```

### Usage

1. **Generate Data**:
   Run `data_generator.py` to generate sample texts and save them to `data/ai-texts.json`.
   ```sh
   python data_generator.py
   ```

2. **Generate Embeddings**:
   Run `embeddings.py` to generate embeddings for the texts and save them to `data/embeddings.json`.
   ```sh
   python embeddings.py
   ```

3. **Run the Streamlit Application**:
   Start the Streamlit app to interactively explore the analyses.
   ```sh
   streamlit run app.py
   ```

## Analyses Included

### Cosine Similarity Heatmap

**Purpose**: To visualize the pairwise similarity between different sections of content.

**How to Read**: The heatmap displays similarity scores between sections. Darker colors indicate higher similarity, while lighter colors indicate lower similarity.

### Hierarchical Clustering

**Purpose**: To identify natural groupings of similar sections.

**How to Read**: The dendrogram shows hierarchical relationships between sections, with closely related sections grouped together.

### K-means Clustering

**Purpose**: To segment sections into clusters based on content similarity.

**How to Read**: The scatter plot shows clusters of content. Points within the same cluster are similar to each other.

### PCA Visualization

**Purpose**: To reduce dimensionality and visualize embeddings in 2D space.

**How to Read**: The scatter plot shows the distribution of sections in a reduced dimension space, with similar sections positioned closely together.

### t-SNE Visualization

**Purpose**: To reduce dimensionality and visualize embeddings in 2D space, preserving local similarities.

**How to Read**: The scatter plot shows the local similarity of sections, with similar sections positioned closely together.

## Understanding the Analyses

By using these embedding techniques, you can see how similar different pieces of content are. This helps in understanding the distribution of content on a site and spotting outliers. Outliers can indicate content that is irrelevant to the site or potentially problematic, especially in the context of media partnerships. For example, entering a media partnership with a site that has no information about betting could be problematic. These visualizations provide a clear way to assess content similarity and relevance, helping you avoid penalties from search engines like Google.

### Example Analyses

- **Content Similarity**: Use the cosine similarity heatmap to ensure that all content aligns well with the site's overall theme.
- **Content Distribution**: Use PCA and t-SNE visualizations to understand how content is distributed across the site.
- **Outlier Detection**: Use hierarchical clustering to spot content that stands out as significantly different from the rest.

By leveraging these tools, you can ensure that your content strategy aligns with best practices and avoids penalties, while also gaining insights into the structure and relevance of your site's content.
