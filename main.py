import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

# 1. Läs data
train_x = np.loadtxt("SmarterML_Training.Input")
train_y = np.loadtxt("SmarterML_Training.Label", dtype=int)
eval_x  = np.loadtxt("SmarterML_Eval.Input")

# 2. Skapa modeller
models = [
    ("log_reg", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ("random_forest", RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample")),
    ("hist_gb", HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)),
]

# 3. Testa modeller (cross-validation)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models:
    scores = cross_validate(model, train_x, train_y, cv=cv, scoring="roc_auc")
    mean_auc = scores["test_score"].mean()
    results.append((name, model, mean_auc))
    print(f"{name}: ROC-AUC = {mean_auc:.4f}")

# 4. Välj bästa modellen
best_name, best_model, best_auc = max(results, key=lambda x: x[2])
print(f"\nBästa modellen: {best_name} (ROC-AUC {best_auc:.4f})")

# 5. Träna bästa modellen och gör prediktioner
best_model.fit(train_x, train_y)
eval_prob = best_model.predict_proba(eval_x)[:, 1]
eval_pred = (eval_prob >= 0.5).astype(int)

print("\nFörsta 10 prediktionerna:", eval_pred[:10])
