# Assignment

This project trains a binary classifier for the Smarter ML cryptocurrency-investor prediction task.

The script in [main.py](/Users/bjornholmgren/Library/Mobile%20Documents/com~apple~CloudDocs/Documents/Todo/ML/Assignment%20-%20Default%20final%20project/Assignment/main.py) does four things:

1. Loads the provided training and evaluation files.
2. Compares a small set of classifier candidates with stratified cross-validation.
3. Selects the best model by mean ROC-AUC.
4. Fits the best model on the full training set and writes predictions for the 1750 evaluation rows.

All file names and settings are defined directly in the script, so there are no command-line arguments to pass.

## Run

From the project directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

## Outputs

Running the script creates an `artifacts/` directory with:

- `model_report.json`: dataset summary, cross-validation results, and the selected model.
- `eval_predictions.txt`: one predicted label (`0` or `1`) per row of `SmarterML_Eval.Input`.
- `eval_probabilities.txt`: predicted probability of class `1` for each evaluation row.
- `best_model.pkl`: serialized fitted model.

## Models Compared

The current candidate pool includes:

- Logistic regression with standardization.
- Random forest.
- Histogram gradient boosting.

This keeps the project simpler while still comparing one linear model and two tree-based models.

On the current dataset, the best cross-validated model is histogram gradient boosting with mean ROC-AUC of about `0.923`.
