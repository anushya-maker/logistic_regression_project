import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================
# 1. Custom Logistic Regression Implementation
# ==========================================

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    # Sigmoid Activation Function
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Binary Cross-Entropy Cost Function
    def _compute_cost(self, y_true, y_pred):
        # Adding a small epsilon to prevent log(0) errors
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = len(y_true)
        cost = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    # Training the model (Gradient Descent)
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters (weights as zeros)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            # 1. Linear model (z = wx + b)
            linear_model = np.dot(X, self.weights) + self.bias
            
            # 2. Apply activation (sigmoid)
            y_predicted = self._sigmoid(linear_model)

            # 3. Compute Gradients
            # Derivative with respect to weights
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # Derivative with respect to bias
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 4. Update Parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Optional: Record cost every 100 iterations to track progress
            if i % 100 == 0:
                cost = self._compute_cost(y, y_predicted)
                self.cost_history.append(cost)

    # Prediction function
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # Convert probabilities to class labels (0 or 1) using 0.5 threshold
        y_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_class)

# Helper function to calculate metrics manually for the custom model
def calculate_metrics(y_true, y_pred):
    # True Positives, False Positives, etc.
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return accuracy, precision, recall

# ==========================================
# 2. Main Execution
# ==========================================

if __name__ == "__main__":
    # --- Step 1: Data Generation ---
    print("Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_classes=2,
        random_state=42, # Ensures reproducibility
        n_informative=4, 
        n_redundant=0
    )

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Step 2 & 3: Train Custom Model ---
    print("\nTraining Custom Logistic Regression Model...")
    custom_model = LogisticRegressionFromScratch(learning_rate=0.01, iterations=1000)
    custom_model.fit(X_train, y_train)

    print(f"Final Weights: {custom_model.weights}")
    print(f"Final Bias: {custom_model.bias}")

    # --- Step 4: Evaluate Custom Model ---
    custom_preds = custom_model.predict(X_test)
    c_acc, c_prec, c_rec = calculate_metrics(y_test, custom_preds)

    print("\n--- Custom Implementation Results ---")
    print(f"Accuracy:  {c_acc:.4f}")
    print(f"Precision: {c_prec:.4f}")
    print(f"Recall:    {c_rec:.4f}")

    # --- Step 5: Compare with Scikit-Learn ---
    print("\nTraining Scikit-Learn Model for Comparison...")
    # Note: sklearn uses regularization by default. Setting penalty=None to match our vanilla implementation closer,
    # though slight differences are expected due to solvers.
    sklearn_model = LogisticRegression(penalty=None, random_state=42) 
    sklearn_model.fit(X_train, y_train)
    sklearn_preds = sklearn_model.predict(X_test)

    sk_acc = accuracy_score(y_test, sklearn_preds)
    sk_prec = precision_score(y_test, sklearn_preds)
    sk_rec = recall_score(y_test, sklearn_preds)

    print("\n--- Scikit-Learn Results ---")
    print(f"Accuracy:  {sk_acc:.4f}")
    print(f"Precision: {sk_prec:.4f}")
    print(f"Recall:    {sk_rec:.4f}")

    # --- Comparison Summary ---
    print("\n--- Comparison ---")
    print(f"Accuracy Difference: {abs(c_acc - sk_acc):.4f}")
