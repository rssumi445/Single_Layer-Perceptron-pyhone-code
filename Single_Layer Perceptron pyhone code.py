class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=50):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict_one(xi)
                error = target - prediction

                # update weights and bias
                for i in range(n_features):
                    self.weights[i] += self.lr * error * xi[i]
                self.bias += self.lr * error

    def predict_one(self, x):
        total = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1 if total >= 0 else 0

    def predict(self, X):
        return [self.predict_one(x) for x in X]


# Example usage (AND gate)
if __name__ == "__main__":
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y = [0, 0, 0, 1]

    p = Perceptron(learning_rate=0.1, epochs=50)
    p.fit(X, y)

    print("Weights:", p.weights)
    print("Bias:", p.bias)
    print("Predictions:", p.predict(X))
