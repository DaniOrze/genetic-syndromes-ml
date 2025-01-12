import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from itertools import product

BASE_PATH = os.path.join("src", "metrics")

os.makedirs(BASE_PATH, exist_ok=True)

def evaluate_tsne_parameters(df, perplexities, learning_rates, max_iter_list):
    """
    Evaluates t-SNE performance across multiple parameter combinations.
    
    Args:
        df (pandas.DataFrame): DataFrame containing image embeddings and syndrome IDs.
        perplexities (list): List of perplexity values to test.
        learning_rates (list): List of learning rates to test.
        max_iter_list (list): List of maximum iterations to test.
    
    Returns:
        results (list): List of parameter combinations and corresponding silhouette scores.
        best_params (tuple): Best parameter combination with the highest silhouette score.
    """
    embeddings = np.stack(df['image_data'].values)
    best_params = None
    best_score = -1
    results = []
    
    for perplexity, learning_rate, max_iter in product(perplexities, learning_rates, max_iter_list):
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=42
        )
        embeddings_2d = tsne.fit_transform(embeddings)
        df['tsne_1'] = embeddings_2d[:, 0]
        df['tsne_2'] = embeddings_2d[:, 1]
        score = silhouette_score(embeddings_2d, df['syndrome_id'])
        results.append((perplexity, learning_rate, max_iter, score))
        
        if score > best_score:
            best_score = score
            best_params = (perplexity, learning_rate, max_iter)
        print(f"Params: perplexity={perplexity}, learning_rate={learning_rate}, n_iter={max_iter}, silhouette_score={score:.4f}")
    
    print("\nBest params:")
    print(f"Perplexity: {best_params[0]}, Learning Rate: {best_params[1]}, Iterações: {best_params[2]}, Silhouette Score: {best_score:.4f}")
    return results, best_params
    
def visualize_data(df):
    """
    Creates and saves various visualizations to analyze t-SNE results and data distributions.
    
    Args:
        df (pandas.DataFrame): DataFrame with t-SNE dimensions and other data.
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='tsne_1',
        y='tsne_2',
        hue='syndrome_id',
        palette='tab10',
        data=df,
        legend="full",
        alpha=0.7
    )
    plt.title("Embeddings Visualization t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title='Syndrome', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, "t-SNE_embeddings.png"))
    plt.show()
    
    df.hist(figsize=(15, 8), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle("Distributions of Variables")
    plt.savefig(os.path.join(BASE_PATH, "variables.png"))
    plt.show()
    
    df['image_data'].apply(lambda x: x[0]).hist(bins=20, figsize=(12, 6), color='skyblue', edgecolor='black')
    plt.title("Distribution of the First Dimension of Embedding")
    plt.savefig(os.path.join(BASE_PATH, "first_dimension_distribution.png"))
    plt.show()
    
    sns.pairplot(df[['tsne_1', 'tsne_2', 'syndrome_id']], hue='syndrome_id', palette='tab10')
    plt.suptitle("Relationships between t-SNE Dimensions and Syndrome")
    plt.savefig(os.path.join(BASE_PATH, "relationships.png"))
    plt.show()

    df['image_data_mean'] = df['image_data'].apply(np.mean)
    corr = df[['tsne_1', 'tsne_2', 'image_data_mean']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(BASE_PATH, "cm.png"))
    plt.show()
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='syndrome_id', y='tsne_1', data=df)
    plt.title("Distribution of the First Dimension t-SNE by Syndrome")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(BASE_PATH, "first_dimension_tsne.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='syndrome_id', y='tsne_2', data=df)
    plt.title("Distribution of the Second Dimension t-SNE by Syndrome")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(BASE_PATH, "second_dimension_tsne.png"))
    plt.show()

def visualize_tsne_results(results):
    """
    Visualizes t-SNE results in a table and as a heatmap of silhouette scores.
    
    Args:
        results (list): List of parameter combinations and silhouette scores.
    """
    df_results = pd.DataFrame(results, columns=['Perplexity', 'Learning Rate', 'Iterations', 'Silhouette Score'])

    print("\nt-SNE Results Table:")
    print(df_results)

    df_results.to_csv(os.path.join(BASE_PATH, "tsne_results.csv"), index=False)

    pivot_table = df_results.pivot_table(index='Perplexity', columns='Learning Rate', values='Silhouette Score', aggfunc='mean')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.4f', cbar_kws={'label': 'Silhouette Score'})
    plt.title("Heat Plot - Silhouette Score vs t-SNE Parameters")
    plt.xlabel("Learning Rate")
    plt.ylabel("Perplexity")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, "tsne_heatmap.png"))
    plt.show()