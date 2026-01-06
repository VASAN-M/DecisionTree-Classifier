import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

# Load dataset (relative path)
df = pd.read_csv("dataset/diabetes.csv")

# Basic exploration
print(df.head())
print(df.info())
print(df.describe())
print("Null values:\n", df.isnull().sum())
print("Shape:", df.shape)

# Group analysis
age_pregnancy = df.groupby("Age")["Pregnancies"].sum()

age_pregnancy.plot(kind="bar", title="Total Pregnancies by Age")
plt.show()

# Features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction & evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Decision tree visualization
tree_graph = graphviz.Source(
    export_graphviz(
        model,
        feature_names=X.columns,
        class_names=["Non-Diabetic", "Diabetic"],
        filled=True
    )
)

tree_graph
