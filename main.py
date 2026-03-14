import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

# 1. Läs data
train_x = np.loadtxt("SmarterML_Training.Input")
train_y = np.loadtxt("SmarterML_Training.Label", dtype=int)
eval_x  = np.loadtxt("SmarterML_Eval.Input")

# 2. Lista av modeller
models = [
    ("log_reg", LogisticRegression(max_iter=5000)),
    ("rf", RandomForestClassifier()),
    ("hgb", HistGradientBoostingClassifier()),
]

# 3. Testa modellerna
results = []
for name, model in models:
    scores = cross_validate(model, train_x, train_y, cv=5, scoring="roc_auc")
    mean_auc = scores["test_score"].mean()
    results.append((name, model, mean_auc))
    print(name, mean_auc)

# 4. Välj bästa modellen
best_name, best_model, best_auc = max(results, key=lambda x: x[2])
print("Bästa modellen:", best_name)

# 5. Träna bästa modellen och förutsäg eval-data
best_model.fit(train_x, train_y)
eval_pred = best_model.predict(eval_x)
print("Prediktioner:", eval_pred[:10])
