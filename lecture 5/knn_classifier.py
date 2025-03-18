import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


def evaluate_knn(X_train, X_test, y_train, y_test, k_max=20):
    """
    Trains and evaluates KNN classifiers for different values of k.

    Parameters:
    - X_train, X_test: Scaled training and test features
    - y_train, y_test: Corresponding labels
    - k_max: The maximum value of k to evaluate

    Returns:
    - k_list: List of tested k values
    - cv_scores: List of accuracy scores for each k
    - best_k: The optimal k value
    - best_score: The highest accuracy achieved
    """
    k_list = list(range(1, k_max + 1))
    cv_scores = []

    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)  # Model evaluation
        cv_scores.append(score)

    best_index = np.argmax(cv_scores)
    best_k = k_list[best_index]
    best_score = cv_scores[best_index]

    return k_list, cv_scores, best_k, best_score


def plot_knn_results(k_list, cv_scores, best_k, best_score):
    """
    Plots misclassification error for different k values in KNN.

    Parameters:
    - k_list: List of k values
    - cv_scores: Corresponding accuracy scores
    - best_k: The best k value
    - best_score: The highest accuracy achieved

     Returns:
    - None: Plots misclassification error for different k values in KNN.
    """
    MSE = [1 - x for x in cv_scores]  # Misclassification error

    plt.figure(figsize=(8, 6))
    plt.plot(k_list, MSE, marker='o', linestyle='-',
             markersize=5, label="Misclassification Error")
    plt.scatter(best_k, 1 - best_score, color='red',
                s=100, label=f'Best K = {best_k}')
    plt.title('The optimal number of neighbors', fontsize=15)
    plt.xlabel('Number of Neighbors K', fontsize=12)
    plt.ylabel('Misclassification Error', fontsize=12)
    plt.legend()
    plt.show()


def plot_knn_decision_boundary(X_train, y_train, X, y, k):
    """
    Visualizes the decision boundary for a KNN classifier.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X: Data features (for plotting)
    - y: Data labels (for plotting)
    - k: Number of neighbors for KNN

    Returns:
    - None: Plots the decision boundary and the data points.
    """
    h = .05  # Step size in mesh grid
    cmap_bold = ListedColormap(
        ['blue', '#FFFF00', 'black', 'green'])

    # Initialize the classifier with best_k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Create mesh grid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict labels for the entire grid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap_bold)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o',
                s=30, cmap=cmap_bold)  # Scatter plot
    plt.title(f"KNN Classifier Decision Boundary (k={k})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
