import numpy as np
from sklearn.manifold import TSNE
from dataProcessing import load_data, flatten_data, preprocess_data
from dataVisualization import evaluate_tsne_parameters, visualize_data, visualize_tsne_results
from classification import knn_classification
from metricsAndEvaluation import generate_roc_auc_curves, summarize_metrics

def main():
    file_path = "data/mini_gm_public_v0.1.p"

    data = load_data(file_path)

    df = flatten_data(data)
    df = preprocess_data(df)
    
    print("\nFlattened data:")
    print(df.head())
    
    print("\nCheck null values:")
    print(df.isnull().sum())
    
    print("\nCorrect image vector size:")
    lengths = df['image_data'].apply(len)
    print(lengths.value_counts())

    print("\nChecking for duplicate values:")
    print(df[['syndrome_id', 'subject_id', 'image_id']].duplicated().sum())
    
    print("\nGeneral information:")
    print(df.info())
    
    print("\nData descriptions:")
    print(df.describe())
    
    print("\nDistribution of images by syndrome:")
    print(df['syndrome_id'].value_counts())
    
    print("\nNumber of subjects per syndrome:")
    subject_counts = df.groupby('syndrome_id')['subject_id'].nunique()
    print(subject_counts)

    print("\nt-SNE Parameter Test:")
    perplexities = [5, 10, 30, 50]
    learning_rates = [10, 100, 500]
    max_iter_list = [300, 500, 1000]

    results, best_params = evaluate_tsne_parameters(df, perplexities, learning_rates, max_iter_list)
    
    visualize_tsne_results(results)

    print("\nRunning t-SNE with the best parameters:")
    tsne = TSNE(
        n_components=2,
        perplexity=best_params[0],
        learning_rate=best_params[1],
        max_iter=best_params[2],
        random_state=42
    )
    embeddings = np.stack(df['image_data'].values)
    embeddings_2d = tsne.fit_transform(embeddings)
    df['tsne_1'] = embeddings_2d[:, 0]
    df['tsne_2'] = embeddings_2d[:, 1]
    
    print("\nUseful charts:")
    visualize_data(df)
    
    print("\nClassification with KNN:")
    results = knn_classification(df)
    
    generate_roc_auc_curves(results)

    summary_df = summarize_metrics(results)
    print("\nMetrics summary:")
    print(summary_df)
    
if __name__ == "__main__":
    main()
