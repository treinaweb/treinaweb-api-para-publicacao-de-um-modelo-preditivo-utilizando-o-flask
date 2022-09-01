from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

iris = load_iris()

X = iris.data
y = iris.target

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=10)

clf.fit(X_treino, y_treino)

previsto = clf.predict(X_teste)

print(accuracy_score(y_teste, previsto))

with open('modelo/treino.pkl', 'wb') as modelo:
    pickle.dump(clf, modelo)