import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print(df.head())
print(df["species"].value_counts())

sns.pairplot(df.drop(columns="target"), hue="species")

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("train:", X_train.shape, " test:", X_test.shape)

model = KNeighborsClassifier(n_neighbors=5)  # 1. create
model.fit(X_train, y_train)  # 2. learn
y_pred = model.predict(X_test)  # 3. predict

print("Test accuracy:", accuracy_score(y_test, y_pred))

# Predict a brand-new flower
new_flower = pd.DataFrame([[5.0, 3.4, 1.5, 0.2]], columns=X.columns)
print("Prediction:", iris.target_names[model.predict(new_flower)[0]])
print("Probabilities:", model.predict_proba(new_flower).round(2))
