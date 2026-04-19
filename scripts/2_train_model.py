"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_model():
    
    #Trains a RandomForestClassifier on the pose data and saves the model.
    
    # --- CONFIGURATION ---
    DATA_CSV_PATH = '../data/pose_data.csv'
    MODEL_DIR = '../models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')

    # Create the models directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load the dataset
    if not os.path.exists(DATA_CSV_PATH):
        print(f"Error: Data file not found at {DATA_CSV_PATH}")
        print("Please run the data collection script (1_collect_data.py) first.")
        return
        
    print("Loading dataset...")
    df = pd.read_csv(DATA_CSV_PATH)
    
    # Check if the dataframe is empty
    if df.empty:
        print("Error: The CSV file is empty. No data to train on.")
        return

    print("Dataset loaded successfully.")
    print(df.head())
    print("\nClass distribution:")
    print(df['label'].value_counts())

    # 2. Prepare the data
    X = df.drop('label', axis=1) # Features
    y = df['label']              # Target variable

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train the RandomForest Classifier
    print("\nTraining RandomForest Classifier...")
    # Using class_weight='balanced' to handle potential class imbalance
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_classifier.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Evaluate the model
    print("\nEvaluating model performance...")
    y_pred = rf_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Save the trained model
    print(f"\nSaving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(rf_classifier, f)
    print("Model saved successfully.")

if __name__ == '__main__':
    train_model()

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# --- Paths ---
DATA_CSV_PATH = '../data/pose_data.csv'
MODEL_DIR = '../models'
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load Data
print("Loading data from CSV...")
try:
    df = pd.read_csv(DATA_CSV_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_CSV_PATH}")
    print("Please run '1_collect_data.py' first to generate the data.")
    exit()

if df.empty:
    print("Error: The data file is empty. No data to train on.")
    exit()

print(f"Data loaded. Shape: {df.shape}")
print(f"Class distribution:\n{df['label'].value_counts(normalize=True)}")

# 2. Define Features (X) and Target (y)
X = df.drop('label', axis=1)
y = df['label']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# 4. Train Model
print("Training RandomForestClassifier...")

# --- UPDATED MODEL ---
# We've made two important changes:
# 1. class_weight='balanced': Fixes the "imbalance" problem (if you have 95% normal and 5% abnormal data).
# 2. max_depth=10: "Prunes" the trees to prevent them from memorizing the data (overfitting).
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    max_depth=10,  # <-- ADDED: Prevents overfitting
    class_weight='balanced' # <-- ADDED: Handles imbalanced data
)
# ---------------------

model.fit(X_train, y_train)
print("Model training complete.")

# 5. Evaluate Model
print("Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Abnormal (1)'])

print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

# 6. Save Model
print(f"Saving model to {MODEL_PATH}...")
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully. You can now run 'app.py'.")

'''
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb  # <-- Changed
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'pose_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl') # We can keep the same name

def train_model():
    """Trains the XGBoost model and saves it to a file."""
    
    print(f"Loading data from {DATA_CSV_PATH}...")
    try:
        df = pd.read_csv(DATA_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_CSV_PATH}")
        print("Please run 'scripts/1_collect_data.py' first.")
        return

    if df.empty:
        print("Error: The data file is empty. No data to train on.")
        return

    print(f"Data loaded. Total frames: {len(df)}")
    class_counts = df['label'].value_counts()
    print("Class distribution:\n", class_counts)

    X = df.drop('label', axis=1)
    y = df['label']

    if len(y.unique()) < 2:
        print("Error: The dataset contains only one class. Cannot train a classifier.")
        print("Please add data for both 'normal' (0) and 'abnormal' (1) classes.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- XGBoost specific: Calculate scale_pos_weight for imbalance ---
    # This is the recommended way to handle imbalanced classes in XGBoost
    train_class_counts = y_train.value_counts()
    if train_class_counts.get(1, 0) == 0:
        print("Error: Training data has no 'abnormal' (1) samples. Stopping.")
        return
        
    scale_pos_weight = train_class_counts[0] / train_class_counts[1]
    print(f"Calculated scale_pos_weight for imbalanced data: {scale_pos_weight:.2f}")

    # --- THIS IS THE UPDATED MODEL ---
    # 1. We now initialize the model *inside* the function.
    # 2. We pass 'scale_pos_weight' here during initialization.
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=5,           # Tuned to prevent overfitting
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight  # <-- The fix is here
    )
    # --- END OF UPDATE ---

    print(f"Starting model training (XGBoost)...")
    
    # We no longer pass the weight argument here
    model.fit(X_train, y_train)
    
    print("Model training complete.")

    # Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\nXGBoost model successfully trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()

'''