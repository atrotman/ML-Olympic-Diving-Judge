import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load preprocessed data
df = pd.read_json('Model building/Preprocessing/preprocessed_data.json')

# Drop irrelevant features
X = df.drop(columns=['score', 'video', 'splash_size', 'dive_type', 'position'])
y = df['score']

# Standardize numerical features
numeric_features = X.select_dtypes(include=['number'])
scaler = StandardScaler()
X[numeric_features.columns] = scaler.fit_transform(numeric_features)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "Support Vector Regressor": SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

# Evaluate each model and log the results
for name, model in models.items():
    # Train models
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate models
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Evaluation metrics
    logger.info(f"{name}:")
    logger.info(f"  Mean Squared Error: {mse:.4f}")
    logger.info(f"  Mean Absolute Error: {mae:.4f}")
    logger.info(f"  R² Score: {r2:.4f}")