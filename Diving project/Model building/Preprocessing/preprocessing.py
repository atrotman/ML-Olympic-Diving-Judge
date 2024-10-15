import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Dive_labeling import difficulty_dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
df = pd.read_json('Shortened videos\Frames and labeling\labeled_results_filtered.json')

# Expand splash size into two width and height
df['splash_width'], df['splash_height'] = zip(*df['splash_size'])

# Assign difficulty from imported dictionary
difficulty_mapping = {
    (dive_type, position): value
    for dive_type, subdict in difficulty_dict.items()
    for position, value in subdict.items()
}

# Use vectorized mapping and fill missing values with default
df['difficulty'] = df.set_index(['dive_type', 'position']).index.map(difficulty_mapping).fillna(3.3)

# Select numeric columns excluding score column
numeric_features = df.select_dtypes(include=['number']).drop(columns=['score'])
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Assign scaled values back to the DataFrame
df[numeric_features.columns] = numeric_features_scaled

# Save the preprocessed data to a CSV file
preprocessed_data_path = 'Model building/Preprocessing/preprocessed_data.json'
df.to_json(preprocessed_data_path, orient='records', indent=4)
logger.info(f"Preprocessed data saved to {preprocessed_data_path}")

# Prepare Features (X) and Target (y)
X = df.drop(columns=['score', 'video', 'splash_size'])
y = df['score']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logging summary of the split
logger.info(f"Training features shape: {X_train.shape}")
logger.info(f"Testing features shape: {X_test.shape}")
logger.info(f"Training target shape: {y_train.shape}")
logger.info(f"Testing target shape: {y_test.shape}")
