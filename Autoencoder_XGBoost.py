# Import necessary libraries
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Create output directory if it doesn't exist
output_dir = '/Users/leopard/Desktop/NIBM/ml2/Sign-Language/Output/Autoencoder/XGBoost'
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

# Define the Autoencoder architecture
input_dim = X_train_scaled.shape[1]  # Number of features
encoding_dim = 32  # Size of the encoded representation (latent space)

# Input layer
input_layer = Input(shape=(input_dim,))

# Encoder
encoder = Dense(128, activation='relu')(input_layer)
encoder = Dense(64, activation='relu')(encoder)
encoder = Dense(encoding_dim, activation='relu')(encoder)

# Decoder
decoder = Dense(64, activation='relu')(encoder)
decoder = Dense(128, activation='relu')(decoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

# Autoencoder model
autoencoder = Model(input_layer, decoder)

# Encoder model (to extract encoded features)
encoder_model = Model(input_layer, encoder)

# Compile the Autoencoder
autoencoder.compile(optimizer='adam', loss='mse')  # Mean Squared Error for reconstruction

# Train the Autoencoder
autoencoder.fit(
    X_train_scaled, X_train_scaled,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test_scaled, X_test_scaled)
)

# Extract encoded features for training and test data
X_train_encoded = encoder_model.predict(X_train_scaled)
X_test_encoded = encoder_model.predict(X_test_scaled)

print(f"Original shape: {X_train.shape}, Encoded shape: {X_train_encoded.shape}")

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_encoded, y_train)

# Reindex labels to be contiguous and zero-indexed
unique_classes = np.unique(y_train_smote)
label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
y_train_smote_reindexed = np.array([label_mapping[label] for label in y_train_smote])
y_test_reindexed = np.array([label_mapping[label] for label in y_test])

# Update num_classes
num_classes = len(unique_classes)
print(f"Number of classes after reindexing: {num_classes}")
print(f"Reindexed class distribution: {np.bincount(y_train_smote_reindexed)}")

# Train XGBoost model
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob',
    num_class=num_classes,  # Use updated num_classes
    tree_method='hist',  # Faster algorithm
    eval_metric=['mlogloss', 'merror'],  # Move eval_metric here
    use_label_encoder=False,
    gamma=0.1,  # Minimum loss reduction for partition
    min_child_weight=3,  # Minimum sum of instance weight in a child
    subsample=0.6,  # Subsample ratio of training instances
    colsample_bytree=0.6,  # Subsample ratio of columns for each tree
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42
)

# Train the model without early stopping
eval_set = [(X_train_smote, y_train_smote_reindexed), (X_test_encoded, y_test_reindexed)]
xgb_model.fit(
    X_train_smote, y_train_smote_reindexed,
    eval_set=eval_set,
    verbose=True
)

# Create history plots manually
# Plot learning curves
plt.figure(figsize=(12, 10))

# Plot feature importance (weight)
plt.subplot(2, 2, 1)
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='weight', ax=plt.gca())
plt.title('Feature Importance (Weight)')

# Plot feature importance (gain)
plt.subplot(2, 2, 2)
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain', ax=plt.gca())
plt.title('Feature Importance (Gain)')

# Plot feature importance (cover)
plt.subplot(2, 2, 3)
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='cover', ax=plt.gca())
plt.title('Feature Importance (Cover)')

# Plot feature importance (total_gain)
plt.subplot(2, 2, 4)
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='total_gain', ax=plt.gca())
plt.title('Feature Importance (Total Gain)')

plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance_plots.png")
plt.show()

# Get evaluation results from the model's attribute
results = xgb_model.evals_result()

# Save model
xgb_model.save_model(f"{output_dir}/xgboost_model.json")

# Plot learning curves
plt.figure(figsize=(12, 10))

# Plot training and validation loss
plt.subplot(2, 2, 1)
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
plt.plot(x_axis, results['validation_0']['mlogloss'], 'b-', label='Train Loss')
plt.plot(x_axis, results['validation_1']['mlogloss'], 'r-', label='Validation Loss')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.grid(True)

# Plot training and validation error
plt.subplot(2, 2, 2)
plt.plot(x_axis, results['validation_0']['merror'], 'b-', label='Train Error')
plt.plot(x_axis, results['validation_1']['merror'], 'r-', label='Validation Error')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.grid(True)

# Plot feature importance
plt.subplot(2, 2, 3)
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='weight', ax=plt.gca())
plt.title('Feature Importance (Weight)')

# Plot feature importance by gain
plt.subplot(2, 2, 4)
xgb.plot_importance(xgb_model, max_num_features=10, importance_type='gain', ax=plt.gca())
plt.title('Feature Importance (Gain)')

plt.tight_layout()
plt.savefig(f"{output_dir}/learning_curves.png")
plt.show()

# Make predictions
y_pred = xgb_model.predict(X_test_encoded)

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