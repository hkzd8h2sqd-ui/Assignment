from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_TRAIN_INPUT = "SmarterML_Training.Input"
DEFAULT_TRAIN_LABEL = "SmarterML_Training.Label"
DEFAULT_EVAL_INPUT = "SmarterML_Eval.Input"


@dataclass(frozen=True)
class Candidate:
	name: str
	estimator: BaseEstimator


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train and evaluate binary classifiers for the Smarter ML assignment."
	)
	parser.add_argument("--train-input", type=Path, default=Path(DEFAULT_TRAIN_INPUT))
	parser.add_argument("--train-label", type=Path, default=Path(DEFAULT_TRAIN_LABEL))
	parser.add_argument("--eval-input", type=Path, default=Path(DEFAULT_EVAL_INPUT))
	parser.add_argument(
		"--artifacts-dir",
		type=Path,
		default=Path("artifacts"),
		help="Directory where reports, model, and predictions are written.",
	)
	parser.add_argument(
		"--cv-folds",
		type=int,
		default=5,
		help="Number of stratified cross-validation folds.",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed used for CV shuffling and stochastic models.",
	)
	return parser.parse_args()


def load_matrix(path: Path) -> np.ndarray:
	matrix = np.loadtxt(path, dtype=np.float64)
	if matrix.ndim == 1:
		matrix = matrix.reshape(1, -1)
	return matrix


def load_labels(path: Path) -> np.ndarray:
	labels = np.loadtxt(path, dtype=np.int64)
	if labels.ndim != 1:
		labels = labels.reshape(-1)
	unique = set(np.unique(labels).tolist())
	if not unique.issubset({0, 1}):
		raise ValueError(f"Labels must be binary 0/1 values, got {sorted(unique)}")
	return labels


def validate_shapes(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray) -> None:
	if train_x.shape[0] != train_y.shape[0]:
		raise ValueError(
			f"Training inputs and labels disagree: {train_x.shape[0]} rows vs {train_y.shape[0]} labels"
		)
	if train_x.shape[1] != eval_x.shape[1]:
		raise ValueError(
			f"Training and evaluation inputs disagree on feature count: {train_x.shape[1]} vs {eval_x.shape[1]}"
		)


def build_candidates(random_state: int) -> list[Candidate]:
	return [
		Candidate(
			"logistic_regression",
			Pipeline(
				[
					("scale", StandardScaler()),
					(
						"model",
						LogisticRegression(
							C=1.0,
							class_weight="balanced",
							max_iter=5000,
							solver="lbfgs",
						),
					),
				]
			),
		),
		Candidate(
			"random_forest",
			RandomForestClassifier(
				n_estimators=400,
				max_depth=None,
				min_samples_leaf=2,
				class_weight="balanced_subsample",
				random_state=random_state,
				n_jobs=-1,
			),
		),
		Candidate(
			"hist_gradient_boosting",
			HistGradientBoostingClassifier(
				learning_rate=0.05,
				max_depth=6,
				max_leaf_nodes=31,
				min_samples_leaf=10,
				l2_regularization=0.05,
				random_state=random_state,
			),
		),
	]


def evaluate_candidates(
	train_x: np.ndarray,
	train_y: np.ndarray,
	candidates: list[Candidate],
	cv_folds: int,
	random_state: int,
) -> list[dict[str, Any]]:
	cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
	scoring = {
		"roc_auc": "roc_auc",
		"accuracy": "accuracy",
		"balanced_accuracy": "balanced_accuracy",
		"f1": "f1",
	}
	results: list[dict[str, Any]] = []

	for candidate in candidates:
		scores = cross_validate(
			candidate.estimator,
			train_x,
			train_y,
			cv=cv,
			scoring=scoring,
			n_jobs=-1,
			error_score="raise",
		)
		result = {
			"name": candidate.name,
			"estimator": candidate.estimator,
			"roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
			"roc_auc_std": float(np.std(scores["test_roc_auc"])),
			"accuracy_mean": float(np.mean(scores["test_accuracy"])),
			"accuracy_std": float(np.std(scores["test_accuracy"])),
			"balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
			"balanced_accuracy_std": float(np.std(scores["test_balanced_accuracy"])),
			"f1_mean": float(np.mean(scores["test_f1"])),
			"f1_std": float(np.std(scores["test_f1"])),
		}
		results.append(result)

	results.sort(
		key=lambda row: (
			row["roc_auc_mean"],
			row["balanced_accuracy_mean"],
			row["f1_mean"],
			row["accuracy_mean"],
		),
		reverse=True,
	)
	return results


def summarize_training_set(train_y: np.ndarray) -> dict[str, Any]:
	class_counts = np.bincount(train_y, minlength=2)
	positive_rate = float(class_counts[1] / class_counts.sum())
	return {
		"num_samples": int(class_counts.sum()),
		"class_0": int(class_counts[0]),
		"class_1": int(class_counts[1]),
		"positive_rate": positive_rate,
	}


def fit_and_predict(
	best_estimator: BaseEstimator,
	train_x: np.ndarray,
	train_y: np.ndarray,
	eval_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
	best_estimator.fit(train_x, train_y)
	eval_probabilities = best_estimator.predict_proba(eval_x)[:, 1]
	eval_predictions = (eval_probabilities >= 0.5).astype(np.int64)

	train_probabilities = best_estimator.predict_proba(train_x)[:, 1]
	train_predictions = (train_probabilities >= 0.5).astype(np.int64)
	training_metrics = {
		"train_roc_auc": float(roc_auc_score(train_y, train_probabilities)),
		"train_accuracy": float(accuracy_score(train_y, train_predictions)),
		"train_balanced_accuracy": float(balanced_accuracy_score(train_y, train_predictions)),
		"train_f1": float(f1_score(train_y, train_predictions)),
	}
	return eval_predictions, eval_probabilities, training_metrics


def serialize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
	serializable: list[dict[str, Any]] = []
	for row in results:
		cleaned = {key: value for key, value in row.items() if key != "estimator"}
		serializable.append(cleaned)
	return serializable


def save_outputs(
	artifacts_dir: Path,
	report: dict[str, Any],
	best_estimator: BaseEstimator,
	eval_predictions: np.ndarray,
	eval_probabilities: np.ndarray,
) -> None:
	artifacts_dir.mkdir(parents=True, exist_ok=True)

	report_path = artifacts_dir / "model_report.json"
	report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

	predictions_path = artifacts_dir / "eval_predictions.txt"
	np.savetxt(predictions_path, eval_predictions, fmt="%d")

	probabilities_path = artifacts_dir / "eval_probabilities.txt"
	np.savetxt(probabilities_path, eval_probabilities, fmt="%.8f")

	model_path = artifacts_dir / "best_model.pkl"
	with model_path.open("wb") as model_file:
		pickle.dump(best_estimator, model_file)


def main() -> None:
	args = parse_args()

	train_x = load_matrix(args.train_input)
	train_y = load_labels(args.train_label)
	eval_x = load_matrix(args.eval_input)
	validate_shapes(train_x, train_y, eval_x)

	candidates = build_candidates(args.random_state)
	results = evaluate_candidates(
		train_x=train_x,
		train_y=train_y,
		candidates=candidates,
		cv_folds=args.cv_folds,
		random_state=args.random_state,
	)

	best_result = results[0]
	best_estimator = best_result["estimator"]
	eval_predictions, eval_probabilities, training_metrics = fit_and_predict(
		best_estimator=best_estimator,
		train_x=train_x,
		train_y=train_y,
		eval_x=eval_x,
	)

	report = {
		"dataset": {
			**summarize_training_set(train_y),
			"num_features": int(train_x.shape[1]),
			"num_eval_samples": int(eval_x.shape[0]),
		},
		"selection_metric": "roc_auc",
		"best_model": {
			key: value for key, value in best_result.items() if key != "estimator"
		},
		"top_models": serialize_results(results[:5]),
		"training_fit_metrics": training_metrics,
	}

	save_outputs(
		artifacts_dir=args.artifacts_dir,
		report=report,
		best_estimator=best_estimator,
		eval_predictions=eval_predictions,
		eval_probabilities=eval_probabilities,
	)

	print("Top candidates ranked by mean ROC-AUC:")
	for row in results[:5]:
		print(
			f"- {row['name']}: roc_auc={row['roc_auc_mean']:.4f}, "
			f"balanced_accuracy={row['balanced_accuracy_mean']:.4f}, "
			f"f1={row['f1_mean']:.4f}, accuracy={row['accuracy_mean']:.4f}"
		)
	print(f"\nBest model: {best_result['name']}")
	print(f"Artifacts written to: {args.artifacts_dir.resolve()}")


if __name__ == "__main__":
	main()
