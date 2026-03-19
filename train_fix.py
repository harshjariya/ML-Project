import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
df = pd.read_csv("bank-full.csv", sep=";")

# 2. Define features and target
# 'y' is the target column in your bank-full.csv
y = df["y"].map({"yes": 1, "no": 0}) 
X = df.drop("y", axis=1)

# 3. Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# 4. Setup Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

# 5. Create the full Pipeline
pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier())
])

# 6. Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)

# 7. SAVE THE MODEL
joblib.dump(pipeline, "model.pkl")

print("Model saved successfully as model.pkl")