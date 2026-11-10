import matplotlib; matplotlib.use('Agg')
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
print(X.describe().loc[["mean", "std", "min", "max"]].round(2).T)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

raw_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
scaled_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_train, y_train)

print(f"KNN without scaling: {raw_model.score(X_test, y_test):.3f}")
print(f"KNN with scaling:    {scaled_model.score(X_test, y_test):.3f}")
