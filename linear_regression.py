from re import S
from turtle import color
import numpy as np


class LinearRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # compute gradient for weights and bias
            dw = 1/n_samples * np.dot(X.T, (y_pred - y))
            db = 1/n_samples * np.sum((y_pred - y))
            
            # update params
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        
        return y_pred
    
    
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    
    X, y = datasets.make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    regressor = LinearRegression(lr=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    print("MSE: ", mse)
    
    plt.scatter(X_train, y_train, s=10)
    plt.scatter(X_test, y_test, s=10, color="red")
    plt.plot(X, regressor.predict(X), color='black')
    
    plt.show()