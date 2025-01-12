import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE

def top_k_accuracy(y_true, y_pred_prob, k=5):
    """
    Calculates the Top-K accuracy score.
    
    Args:
    y_true (array): True class labels.
    y_pred_prob (array): Predicted class probabilities.
    k (int): Number of top predictions to consider.
    
    Returns:
    float: Top-K accuracy score.
    """
    top_k_correct = 0
    for true_label, probas in zip(y_true, y_pred_prob):
        top_k_predictions = np.argsort(probas)[::-1][:k]
        if true_label in top_k_predictions:
            top_k_correct += 1
    return top_k_correct / len(y_true)

def knn_classification(df):
    """
    Performs K-Nearest Neighbors (KNN) classification with both Cosine and Euclidean metrics,
    using Stratified K-Fold cross-validation and Borderline-SMOTE for oversampling.
    
    Args:
    df (DataFrame): Input data containing features ('tsne_1', 'tsne_2') and target ('syndrome_id').
    
    Returns:
    list: Results of the classification, including metrics for each K value and distance metric.
    """
    X = df[['tsne_1', 'tsne_2']].values
    y = df['syndrome_id'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    smote = BorderlineSMOTE(random_state=42)

    best_k_cosine = None
    best_score_cosine = -1
    best_f1_cosine = -1
    best_auc_cosine = -1
    
    best_k_euclidean = None
    best_score_euclidean = -1
    best_f1_euclidean = -1
    best_auc_euclidean = -1

    results = []

    for k in range(1, 16):
        knn_cosine = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn_euclidean = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

        cosine_scores, f1_cosine_scores, auc_cosine_scores, top_k_cosine_scores = [], [], [], []
        euclidean_scores, f1_euclidean_scores, auc_euclidean_scores, top_k_euclidean_scores = [], [], [], []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            knn_cosine.fit(X_train_resampled, y_train_resampled)
            y_pred_cosine = knn_cosine.predict(X_test)
            y_pred_prob_cosine = knn_cosine.predict_proba(X_test)

            cosine_scores.append(accuracy_score(y_test, y_pred_cosine))
            f1_cosine_scores.append(f1_score(y_test, y_pred_cosine, average='weighted'))
            auc_cosine_scores.append(roc_auc_score(y_test, y_pred_prob_cosine, multi_class='ovr'))
            top_k_cosine_scores.append(top_k_accuracy(y_test, y_pred_prob_cosine, k=5))

            knn_euclidean.fit(X_train_resampled, y_train_resampled)
            y_pred_euclidean = knn_euclidean.predict(X_test)
            y_pred_prob_euclidean = knn_euclidean.predict_proba(X_test)

            euclidean_scores.append(accuracy_score(y_test, y_pred_euclidean))
            f1_euclidean_scores.append(f1_score(y_test, y_pred_euclidean, average='weighted'))
            auc_euclidean_scores.append(roc_auc_score(y_test, y_pred_prob_euclidean, multi_class='ovr'))
            top_k_euclidean_scores.append(top_k_accuracy(y_test, y_pred_prob_euclidean, k=5))

        mean_cosine_score = np.mean(cosine_scores)
        mean_f1_cosine = np.mean(f1_cosine_scores)
        mean_auc_cosine = np.mean(auc_cosine_scores)
        mean_top_k_cosine = np.mean(top_k_cosine_scores)

        mean_euclidean_score = np.mean(euclidean_scores)
        mean_f1_euclidean = np.mean(f1_euclidean_scores)
        mean_auc_euclidean = np.mean(auc_euclidean_scores)
        mean_top_k_euclidean = np.mean(top_k_euclidean_scores)

        results.append((k, mean_cosine_score, mean_euclidean_score, mean_f1_cosine, mean_f1_euclidean,
                        mean_top_k_cosine, mean_top_k_euclidean, mean_auc_cosine, mean_auc_euclidean))

        if mean_cosine_score > best_score_cosine:
            best_score_cosine = mean_cosine_score
            best_k_cosine = k
            best_auc_cosine = mean_auc_cosine
            best_f1_cosine = mean_f1_cosine

        if mean_euclidean_score > best_score_euclidean:
            best_score_euclidean = mean_euclidean_score
            best_k_euclidean = k
            best_auc_euclidean = mean_auc_euclidean
            best_f1_euclidean = mean_f1_euclidean

    print(f"Best k value for Cosine: {best_k_cosine}")
    print(f"Best Score for Cosine: {best_score_cosine:.2f}")
    print(f"Best F1-Score for Cosine: {best_f1_cosine:.2f}")
    print(f"Best AUC for Cosine: {best_auc_cosine:.2f}")
    
    print(f"Best k value for Euclidean: {best_k_euclidean}")
    print(f"Best Score for Euclidean: {best_score_euclidean:.2f}")
    print(f"Best F1-Score for Euclidean: {best_f1_euclidean:.2f}")
    print(f"Best AUC for Euclidean: {best_auc_euclidean:.2f}")

    return results
