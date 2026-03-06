
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns



# ============================================================
# 2) LOAD THE DATA
# ============================================================
df = pd.read_csv("fossil_leaf_dummy_dataset.csv")

print("\n=== HEAD OF DATA ===")
print(df.head())

# print("\n=== INFO ===")
# print(df.info())

print("\n=== DESCRIPTIVE STATS ===")
print(df.describe(include="all"))


# ============================================================
# 3) DATA EXPLORATION / FEATURE ENGINEERING CHECKS
# ============================================================

# ---- Check for missing values ----
print("\n=== MISSING VALUES PER COLUMN ===")
print(df.isnull().sum())



# ---- Check duplicates ----
print("\n=== DUPLICATE ROWS COUNT ===")
print(df.duplicated().sum())


# ---- Check numerical feature distributions ----
print("\n=== NUMERICAL FEATURES ===")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
print(num_cols)



# ---- Detect zero-variance features (useless for ML) ----
print("\n=== ZERO VARIANCE FEATURES ===")
for col in num_cols:
    if df[col].nunique() == 1:
        print(f"⚠ Zero variance: {col}")


# ---- Correlation matrix ----
corr = df[num_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix of Numerical Features")
plt.show()



# 4) CLASS BALANCE ANALYSIS
# ============================================================

target_col = "species"
class_counts = df[target_col].value_counts()
class_percent = class_counts / len(df) * 100

# ---- Imbalance thresholds ----
# Balanced: <20% difference between largest/smallest class
# Moderately imbalanced: 20–50%
# Highly imbalanced: >50%


max_p = class_percent.max()
min_p = class_percent.min()
gap = max_p - min_p


print("\n=== CLASS BALANCE ASSESSMENT ===")
print(f"Largest class: {max_p:.2f}%  | Smallest class: {min_p:.2f}%")
print(f"Gap: {gap:.2f}%")

if gap < 20:
    print("→ Dataset is BALANCED.")
elif 20 <= gap < 50:
    print("→ Dataset is MODERATELY IMBALANCED.")
else:
    print("→ Dataset is HIGHLY IMBALANCED.")



# ============================================================
# 5) DROP THE LAST ROW FOR INFERENCE LATER
# ============================================================

df_train = df.iloc[:-1].copy()
df_infer = df.iloc[-1:].copy()


# ============================================================
# 6) PREPARE DATA FOR TRAINING
# ============================================================

X = df_train.drop(columns=[target_col, "id"])  # drop non-features and highly correlated features
y = df_train[target_col]

# ---- Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y  # maintains class ratio
)


# ============================================================
# 7) TRAIN RANDOM FOREST CLASSIFIER
# ============================================================

# ---- Parameter justification ----
# n_estimators=1000:
#   More trees → lower variance and more stable predictions on tabular data.
#   Fossil morphology features have nonlinear interactions → RF benefits from many trees.
#
# max_depth=None:
#   Allows trees to grow fully. Since data is synthetic and noise-free, deep trees won't overfit badly.
#
# max_features="sqrt":
#   Standard for classification. Prevents trees from becoming identical → improves generalization.
#
# class_weight=None:
#   Dataset is balanced; no need to weigh classes.

#
# n_jobs=-1:
#   Use all CPU cores for speed.
#
# random_state=42:
#   Ensures reproducibility


rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    max_features="sqrt",
    class_weight=None,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)


# ============================================================
# 8) MODEL EVALUATION USING SCIENTIFICALLY STANDARD METRICS
# ============================================================

y_pred = rf.predict(X_test)


# Accuracy:
#   Percentage of correct predictions. Basic but widely used.
acc = accuracy_score(y_test, y_pred)

# Macro-F1:
#   Uses F1 per class, then averages → treats all classes equally.
f1_macro = f1_score(y_test, y_pred, average="macro")

# Weighted-F1:
#   F1 weighted by class frequency → correct when classes slightly imbalanced.
f1_weighted = f1_score(y_test, y_pred, average="weighted")


print("\n=== MODEL PERFORMANCE ===")
print(f"Accuracy:       {acc:.4f}")
print(f"F1 Macro:       {f1_macro:.4f}")
print(f"F1 Weighted:    {f1_weighted:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))


# ---- Feature Importance ----
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 10))
importances.head(20).plot(kind="barh")
plt.title("Top 20 Feature Importances")
plt.gca().invert_yaxis()
plt.show()


# 9) INFERENCE EXAMPLE USING LAST ROW
# ============================================================

X_infer = df_infer.drop(columns=[target_col, "id"])
prediction = rf.predict(X_infer)[0]
proba = rf.predict_proba(X_infer)[0]

print("\n=== INFERENCE ON LAST ROW ===")
print("Predicted class:", prediction)

# Show probability distribution across classes
proba_df = pd.DataFrame({
    "class": rf.classes_,
    "probability": proba
}).sort_values("probability", ascending=False)

print("\n=== PREDICTION PROBABILITIES ===")
print(proba_df)
