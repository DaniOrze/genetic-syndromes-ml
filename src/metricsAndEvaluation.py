import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

BASE_PATH = os.path.join("src", "metrics")

os.makedirs(BASE_PATH, exist_ok=True)

def generate_roc_auc_curves(results):
    """
    Generate and save ROC AUC curves for Cosine and Euclidean metrics across different k values.
    
    Args:
        results (list): A list of tuples containing metrics for each k value. Each tuple should have the following format:
                        (k, ..., auc_cosine, auc_euclidean).
    
    Steps:
        1. Extract k values and AUC scores for Cosine and Euclidean metrics.
        2. Compute mean AUC for both metrics.
        3. Plot the AUC scores against k values.
        4. Save the plot to the BASE_PATH directory and display it.
    """
    ks, _, _, _, _, _, _, auc_cosine_list, auc_euclidean_list = zip(*results)
    mean_auc_cosine = np.mean(auc_cosine_list)
    mean_auc_euclidean = np.mean(auc_euclidean_list)
    plt.figure(figsize=(10, 6))
    plt.plot(ks, auc_cosine_list, label=f'Cosine AUC (avg: {mean_auc_cosine:.2f})', marker='o')
    plt.plot(ks, auc_euclidean_list, label=f'Euclidean AUC (avg: {mean_auc_euclidean:.2f})', marker='o')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("AUC")
    plt.title("ROC AUC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(os.path.join(BASE_PATH, "roc_auc_comparison.png"))
    plt.show()

def summarize_metrics(results):
    """
    Summarize metrics from the results and save them to a CSV file.
    
    Args:
        results (list): A list of tuples containing metrics for each k value. Each tuple should have the following format:
                        (k, acc_cosine, acc_euclidean, f1_cosine, f1_euclidean, 
                        top_k_cosine, top_k_euclidean, auc_cosine, auc_euclidean).
    
    Steps:
        1. Extract metrics from the results and store them in a dictionary.
        2. Convert the dictionary into a DataFrame.
        3. Save the DataFrame as a CSV file in the BASE_PATH directory.
        4. Return the DataFrame for further analysis.
    """
    summary_data = []
    for (k, acc_cosine, acc_euclidean, f1_cosine, f1_euclidean, top_k_cosine, top_k_euclidean, auc_cosine, auc_euclidean) in results:
        summary_data.append({
            "k": k,
            "Cosine Accuracy": acc_cosine,
            "Euclidean Accuracy": acc_euclidean,
            "Cosine F1-Score": f1_cosine,
            "Euclidean F1-Score": f1_euclidean,
            "Cosine Top-k Accuracy": top_k_cosine,
            "Euclidean Top-k Accuracy": top_k_euclidean,
            "Cosine AUC": auc_cosine,
            "Euclidean AUC": auc_euclidean,
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df["Cosine AUC"] = summary_df["Cosine AUC"].astype(float)
    summary_df["Euclidean AUC"] = summary_df["Euclidean AUC"].astype(float)

    summary_df.to_csv(os.path.join(BASE_PATH, "metrics_summary.csv"), index=False)

    return summary_df
