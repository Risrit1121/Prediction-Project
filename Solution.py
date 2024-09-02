import pandas as pd

# Load the training dataset
train_df = pd.read_csv('Disease_train.csv')

# Load the test dataset
test_df = pd.read_csv('Disease_test.csv')

# Display the first few rows of the training dataset
print("Training Dataset:")
print(train_df.head())

# Display the first few rows of the test dataset
print("\nTest Dataset:")
print(test_df.head())

from sklearn.preprocessing import StandardScaler

# Separate features and target variable for training data
X_train = train_df.drop(columns=['patient_id', 'diagnosis'])
y_train = train_df['diagnosis']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

from sklearn.model_selection import train_test_split

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Print the shapes of training and validation sets
print("Training set shape:", X_train_split.shape, y_train_split.shape)
print("Validation set shape:", X_val_split.shape, y_val_split.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model on the training data
log_reg.fit(X_train_split, y_train_split)

# Predict probabilities on the validation set
y_val_pred_lr = log_reg.predict_proba(X_val_split)[:, 1]

# Calculate ROC-AUC score
roc_auc_lr = roc_auc_score(y_val_split, y_val_pred_lr)
print("ROC-AUC score for Logistic Regression:", roc_auc_lr)

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)

# Train the model on the training data
rf_clf.fit(X_train_split, y_train_split)

# Predict probabilities on the validation set
y_val_pred_rf = rf_clf.predict_proba(X_val_split)[:, 1]

# Calculate ROC-AUC score
roc_auc_rf = roc_auc_score(y_val_split, y_val_pred_rf)
print("ROC-AUC score for Random Forest:", roc_auc_rf)

# Preprocess the test data
X_test = test_df.drop(columns=['patient_id'])
X_test_scaled = scaler.transform(X_test)

# Generate predictions using the Random Forest model
test_predictions = rf_clf.predict_proba(X_test_scaled)[:, 1]

# Create a DataFrame for predictions
submission_df = pd.DataFrame({'patient_id': test_df['patient_id'], 'prediction': test_predictions})

# Save predictions to a CSV file
submission_df.to_csv('student_predictions.csv', index=False)
