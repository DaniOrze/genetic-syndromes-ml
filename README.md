# Genetic Syndromes Analysis - ML 

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)

This project focuses on analyzing and classifying syndrome data using machine learning techniques (KNN), with an emphasis on dimensionality reduction, class balancing, and visualization. The dataset consists of hierarchical data organized into distinct syndromes, which is preprocessed and analyzed to improve model performance and provide actionable insights.

## Features

- Preprocessing of hierarchical data into a flat structure suitable for analysis.
- Dimensionality reduction using t-SNE for cluster visualization.
- Addressing class imbalance with Borderline SMOTE.
- K-Nearest Neighbors (KNN) classification.
- Extensive visualizations, including heatmaps, t-SNE embeddings, and ROC AUC curves.

## Technologies

The project leverages the following Python libraries:

- `numpy` for numerical computations.
- `pandas` for data manipulation.
- `matplotlib` and `seaborn` for data visualization.
- `scikit-learn` for machine learning algorithms.
- `imblearn` for class imbalance handling using SMOTE.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DaniOrze/genetic-syndromes-ml.git
    ```
2. Navigate to the project folder:
    ```bash
   cd genetic-syndromes-ml
   ```
3. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
   ```

## How to Run Locally

1. Navigate to the project folder:
    ```bash
   cd genetic-syndromes-ml
   ```
2. Running the project:
    ```bash
   python src/main.py
   ```

## Output
All generated images and tables will be saved in the metrics folder for further analysis and review.