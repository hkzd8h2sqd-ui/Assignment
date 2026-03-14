import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X = np.loadtxt("SmarterML_Training.Input")
y = np.loadtxt("SmarterML_Training.Label", dtype=int)
X_eval = np.loadtxt("SmarterML_Eval.Input")

models = [
    ("log_reg", LogisticRegression(max_iter=5000)),
    ("rf", RandomForestClassifier()),
    ("hgb", HistGradientBoostingClassifier()),
]

scores = [(name, model, cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()) 
          for name, model in models]

best_name, best_model, best_auc = max(scores, key=lambda x: x[2])
best_model.fit(X, y)
pred = best_model.predict(X_eval)

print("Bästa modellen:", best_name, "ROC-AUC:", best_auc)
print("Prediktioner:", pred[:10])
