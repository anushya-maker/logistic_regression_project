import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================
# 1. Logistic Regression Class
# ==========================================

class MyLogisticRegression:
    def __init__(self, lr=0.01, n_iters=10000):
        # Increased to 10000 iterations as per project requirements
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []

    # Sigmoid function to map inputs to 0-1 range
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Calculating Binary Cross Entropy loss
    def _compute_loss(self, y_true, y_pred):
        # small value to avoid log(0) error
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # init parameters to zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent loop
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_pred)
        # Convert probabilities to 0 or 1
        class_preds = [0 if y <= 0.5 else 1 for y in y_pred]
        return np.array(class_preds)

# ==========================================
# 2. Main Script
# ==========================================

if __name__ == "__main__":
    # Generate data (500 samples, 5 features)
    X, y = make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train My Model ---
    print("Training custom model...")
    my_model = MyLogisticRegression(lr=0.01, n_iters=10000)
    my_model.fit(X_train, y_train)

    # Predictions
    my_preds = my_model.predict(X_test)

    # --- Train Sklearn Model ---
    # Setting penalty to None to match our simple implementation
    print("Training sklearn model...")
    sk_model = LogisticRegression(penalty=None, max_iter=10000, random_state=42)
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)

    # --- Evaluation ---
    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        return acc, prec, rec

    my_acc, my_prec, my_rec = get_metrics(y_test, my_preds)
    sk_acc, sk_prec, sk_rec = get_metrics(y_test, sk_preds)

    print("\n--- RESULTS TO COPY FOR REPORT ---")
    print(f"Custom Accuracy:  {my_acc:.4f}")
    print(f"Custom Precision: {my_prec:.4f}")
    print(f"Custom Recall:    {my_rec:.4f}")
    print("-" * 20)
    print(f"Sklearn Accuracy:  {sk_acc:.4f}")
    print(f"Sklearn Precision: {sk_prec:.4f}")
    print(f"Sklearn Recall:    {sk_rec:.4f}")
    
    print("\n--- LEARNED PARAMETERS (COPY THESE) ---")
    # Using list comprehension to round weights for easier copying
    rounded_weights = [round(w, 4) for w in my_model.weights]
    print(f"Final Weights: {rounded_weights}")
    print(f"Final Bias:    {my_model.bias:.4f}")
