import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Streamlit Config
# -----------------------------

st.set_page_config(
    page_title="AdaBoost Classifier",
    layout="wide"
)

st.title("AdaBoost Classification with Breast Cancer Dataset")

# -----------------------------
# Data Ingestion
# -----------------------------

st.header("1. Data Ingestion")

@st.cache_data
def load_data():

    data = load_breast_cancer(as_frame=True)

    df = data.frame.rename(
        columns={"target": "TARGET"}
    )

    # Save raw dataset
    raw_path = os.path.join(
        RAW_DIR,
        "breast_cancer_dataset.csv"
    )

    df.to_csv(raw_path, index=False)

    # Add missing values randomly
    np.random.seed(42)

    for col in df.columns[:-1]:
        df.loc[df.sample(frac=0.1).index, col] = np.nan

    return df


df = load_data()

st.success("Breast Cancer Dataset Loaded Successfully")

st.dataframe(df, use_container_width=True)

# -----------------------------
# Data Cleaning
# -----------------------------

st.header("2. Data Cleaning")

strategy = st.selectbox(
    "Missing Value Strategy",
    [
        "Mean",
        "Median",
        "Most Frequent",
        "Drop Rows"
    ]
)

df_clean = df.copy()

if strategy == "Drop Rows":

    df_clean = df_clean.dropna()

else:

    fill_map = {
        "Mean": "mean",
        "Median": "median",
        "Most Frequent": "most_frequent"
    }

    imputer = SimpleImputer(
        strategy=fill_map[strategy]
    )

    cols = df_clean.select_dtypes(
        include=np.number
    ).columns

    df_clean[cols] = imputer.fit_transform(
        df_clean[cols]
    )

st.dataframe(df_clean, use_container_width=True)

# Save cleaned dataset
if st.button("Save Cleaned Dataset"):

    cleaned_path = os.path.join(
        CLEAN_DIR,
        "cleaned_breast_cancer_dataset.csv"
    )

    df_clean.to_csv(cleaned_path, index=False)

    st.success("Dataset Saved Successfully")

# -----------------------------
# Load Cleaned Dataset
# -----------------------------

st.header("3. Load Cleaned Dataset")

files = [
    f for f in os.listdir(CLEAN_DIR)
    if "breast_cancer_dataset" in f
]

if not files:

    st.warning("No cleaned dataset found")
    st.stop()

selected_file = st.selectbox(
    "Select Dataset",
    files
)

data = pd.read_csv(
    os.path.join(CLEAN_DIR, selected_file)
)

st.dataframe(data, use_container_width=True)

# -----------------------------
# Sidebar Settings
# -----------------------------

st.sidebar.header("Model Settings")

n_iter = st.sidebar.slider(
    "Random Search Iterations",
    5,
    50,
    20
)

cv_folds = st.sidebar.slider(
    "Cross Validation Folds",
    2,
    10,
    5
)

test_size = st.sidebar.slider(
    "Test Size",
    0.1,
    0.5,
    0.25
)

random_state = st.sidebar.slider(
    "Random State",
    1,
    100,
    42
)

# -----------------------------
# Prepare Data
# -----------------------------

X = data.drop(columns=["TARGET"])
y = data["TARGET"]

# Imputation
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

# -----------------------------
# Model Setup
# -----------------------------

base_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=random_state
)

model = AdaBoostClassifier(
    estimator=base_tree,
    random_state=random_state,
    n_estimators=100
)

# -----------------------------
# Hyperparameter Grid
# -----------------------------

param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.5, 1.0]
}

# -----------------------------
# Randomized Search CV
# -----------------------------

st.header("4. Randomized Search CV Training")

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=n_iter,
    cv=cv_folds,
    scoring="accuracy",
    n_jobs=-1,
    random_state=random_state
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

# -----------------------------
# Predictions
# -----------------------------

pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.success(f"Accuracy Score: {accuracy:.4f}")

st.write("Best Parameters:")
st.write(search.best_params_)

st.write("Best CV Score:")
st.write(search.best_score_)

# -----------------------------
# Classification Report
# -----------------------------

st.header("5. Classification Report")

report = classification_report(
    y_test,
    pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df, use_container_width=True)

# -----------------------------
# Confusion Matrix
# -----------------------------

st.header("6. Confusion Matrix")

cm = confusion_matrix(y_test, pred)

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    ax=ax
)

ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# -----------------------------
# Feature Importance
# -----------------------------

st.header("7. Feature Importance")

importance = pd.DataFrame({

    "Feature": data.drop(
        columns=["TARGET"]
    ).columns,

    "Importance": best_model.feature_importances_

}).sort_values(
    by="Importance",
    ascending=False
)

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    ax=ax2
)

ax2.set_title("Feature Importance")

st.pyplot(fig2)

st.dataframe(
    importance,
    use_container_width=True
)

# -----------------------------
# Save Model
# -----------------------------

st.header("8. Save Model")

model_path = os.path.join(
    MODEL_DIR,
    "adaboost_classifier.pkl"
)

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

st.success(
    f"Model Saved At: {model_path}"
)

# -----------------------------
# Sample Predictions
# -----------------------------

st.header("9. Sample Predictions")

sample = pd.DataFrame({

    "Actual": y_test.values[:10],
    "Predicted": pred[:10]

})

st.dataframe(
    sample,
    use_container_width=True
)