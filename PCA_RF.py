# Import necessary libraries
import numpy as np
import pandas as pd
import os
import sys
import signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# Function to handle exit signals
def signal_handler(sig, frame):
    print("\nProcess interrupted. Exiting gracefully...")
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Create output directory if it doesn't exist
output_dir = '/Users/leopard/Desktop/NIBM/ml2/Sign-Language/Output/PCA/Model2'
os.makedirs(output_dir, exist_ok=True)

# Load the Sign Language MNIST datasets
train_data = pd.read_csv('sign_mnist/sign_mnist_train.csv')
test_data = pd.read_csv('sign_mnist/sign_mnist_test.csv')

# Separate features and labels
y_train = train_data['label'].values
X_train = train_data.drop('label', axis=1).values

y_test = test_data['label'].values
X_test = test_data.drop('label', axis=1).values

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA with 80% variance retention
pca = PCA(n_components=0.8)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Original shape: {X_train.shape}, PCA shape: {X_train_pca.shape}")
print(f"Variance explained: {np.sum(pca.explained_variance_ratio_):.2f}")

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)

# Reindex labels to be contiguous and zero-indexed
unique_classes = np.unique(y_train_smote)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
y_train_smote_reindexed = np.array([label_mapping[label] for label in y_train_smote])
y_test_reindexed = np.array([label_mapping[label] for label in y_test])

# Update num_classes
num_classes = len(unique_classes)
print(f"Number of classes after reindexing: {num_classes}")
print(f"Reindexed class distribution: {np.bincount(y_train_smote_reindexed)}")

# Plot 5 sample images from original data to visualize
plt.figure(figsize=(15, 3))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap='gray')  # Original images are 28x28
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{output_dir}/sample_images.png")
plt.show()

# Initialize the Random Forest model with manually tuned hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=200,  # Number of trees in the forest
    max_depth=20,  # Maximum depth of the tree
    min_samples_split=5,  # Minimum number of samples required to split a node
    min_samples_leaf=2,  # Minimum number of samples required at each leaf node
    max_features='sqrt',  # Number of features to consider at every split
    bootstrap=True,  # Whether bootstrap samples are used
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)

# Train the model
rf_model.fit(X_train_smote, y_train_smote_reindexed)

# Make predictions
y_pred = rf_model.predict(X_test_pca)

# Evaluate model
accuracy = accuracy_score(y_test_reindexed, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("Classification Report:")
report = classification_report(y_test_reindexed, y_pred)
print(report)

# Save classification report to file
with open(f"{output_dir}/classification_report.txt", 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_reindexed, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.show()

# Plot some predictions
plt.figure(figsize=(15, 10))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')  # Original images are 28x28
    true_label = y_test_reindexed[i]
    pred_label = y_pred[i]
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f"True: {true_label}, Pred: {pred_label}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{output_dir}/predictions.png")
plt.show()

# Visualize decision boundaries (for top 2 PCA components)
plt.figure(figsize=(12, 10))
# Use only first two components for visualization
X_test_2d = X_test_pca[:, :2]

# Create a mesh grid
x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Create features for prediction
Z_features = np.c_[xx.ravel(), yy.ravel()]
# Add zeros for remaining features
if X_test_pca.shape[1] > 2:
    Z_features = np.hstack([Z_features, np.zeros((Z_features.shape[0], X_test_pca.shape[1] - 2))])

# Predict class labels for the mesh grid
Z = rf_model.predict(Z_features)
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')

# Plot the test points
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_reindexed, 
                     s=20, edgecolor='k', cmap='rainbow')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Decision Boundaries (First 2 PCA Components)')
plt.colorbar(scatter)
plt.savefig(f"{output_dir}/decision_boundaries.png")
plt.show()