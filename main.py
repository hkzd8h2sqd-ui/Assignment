import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# 1. Läs data
X = np.loadtxt("SmarterML_Training.Input")
y = np.loadtxt("SmarterML_Training.Label", dtype=int)
X_eval = np.loadtxt("SmarterML_Eval.Input")

# -----------------------------
# Logistic Regression
# -----------------------------
log_reg = LogisticRegression(max_iter=5000)
log_reg_auc = cross_val_score(log_reg, X, y, cv=5, scoring="roc_auc").mean()
print("Logistic Regression ROC-AUC:", log_reg_auc)

# -----------------------------
# Random Forest
# -----------------------------
rf = RandomForestClassifier()
rf_auc = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()
print("Random Forest ROC-AUC:", rf_auc)

# -----------------------------
# Hist Gradient Boosting
# -----------------------------
hgb = HistGradientBoostingClassifier()
hgb_auc = cross_val_score(hgb, X, y, cv=5, scoring="roc_auc").mean()
print("Hist Gradient Boosting ROC-AUC:", hgb_auc)

# -----------------------------
# Välj bästa modellen
# -----------------------------
best_auc = max(log_reg_auc, rf_auc, hgb_auc)

if best_auc == log_reg_auc:
    best_name = "Logistic Regression"
    best_model = log_reg
elif best_auc == rf_auc:
    best_name = "Random Forest"
    best_model = rf
else:
    best_name = "Hist Gradient Boosting"
    best_model = hgb

print("\nBästa modellen:", best_name, "ROC-AUC:", best_auc)

# -----------------------------
# Träna bästa modellen och gör prediktioner
# -----------------------------
best_model.fit(X, y)
pred = best_model.predict(X_eval)

# Spara prediktionerna (1750 rader, 0 eller 1)
np.savetxt("SmarterML_Eval_predictions.txt", pred, fmt="%d")

print("Första 10 prediktionerna:", pred[:10])
