
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
iris = load_iris()
X = iris.data          # features
y = iris.target        # labels (0,1,2)

# 2. Train–test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train KNN model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4. Predict on test data
y_pred = knn.predict(X_test)

# 5. Separate correct and incorrect predictions
correct = []
incorrect = []

for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        # (features, true_label)
        correct.append((X_test[i], y_test[i]))
    else:
        # (features, true_label, predicted_label)
        incorrect.append((X_test[i], y_test[i], y_pred[i]))

# 6. Print correct predictions
print("\nCorrect predictions:\n")
for sample, true_label in correct:
    print(
        "Features:", sample,
        "  Actual:", iris.target_names[true_label]
    )

# 7. Print incorrect predictions
print("\nIncorrect predictions:\n")
for sample, true_label, pred_label in incorrect:
    print(
        "Features:", sample,
        "  Actual:", iris.target_names[true_label],
        "  Predicted:", iris.target_names[pred_label]
    )

# 8. Accuracy
acc = accuracy_score(y_test, y_pred)
acc=acc*100
print("\nAccuracy:",acc)

# 9. Confusion matrix (numbers)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (numbers):\n", cm)

# 10. Confusion matrix heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(
    cm, annot=True, fmt="d",
    cmap="Reds",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
