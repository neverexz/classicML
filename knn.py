from cProfile import label
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt( np.sum( (x1-x2)**2  ) )

class KNN:
    
    def __init__(self, k=3, task="classification"):
        
        self.k = k
        self.task = task
        
    def fit(self, X, y):
        
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        
        predicted_labels = [self._predict(x) for x in X]
        return  np.array(predicted_labels)
        
    def _predict(self, x):
        # find distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # get k nearest 
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # get most common value, returned in format ((value, count))
        if self.task == "classification":
            
            most_common = Counter(k_nearest_labels).most_common(1)
            # get value from ((value, count))
            return most_common[0][0]
        
        elif self.task == "regression":
            
            return np.mean(k_nearest_labels)
        
        else:
            
            return ValueError("task must be 'classification' or 'regression'")
    
if __name__ == "__main__":
    from matplotlib.colors import ListedColormap
    from matplotlib import pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    cmap = ListedColormap(["#F0F000", "#F00F00", "#F000F0"])
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    k = 10
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(accuracy(y_test, predictions))

    fig, axes = plt.subplots(2, 2, figsize=(15, 5))

    axes[0, 0].scatter(X[:, 0], X[:, 1], c=y)
    axes[0, 0].set_title("true cls")

    axes[0, 1].scatter(X[:, 0], X[:, 1], c=clf.predict(X))
    axes[0, 1].set_title("predicted cls")
    
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    reg = KNN(k=5, task="regression")
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    print("MSE:", np.mean((predictions - y_test)**2))
    
    axes[1, 0].plot(y_test, marker="o")
    axes[1, 0].set_title("true cls")
    
    axes[1, 1].plot(predictions, marker="x")
    axes[1, 1].set_title("predicted cls")
    
    plt.tight_layout()
    plt.show()
