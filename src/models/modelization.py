from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_curve
from sklearn.model_selection import cross_validate, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from src.visualization.visu_text import print_title, print_col, print_end


def get_score_classification(y, prediction):
    accuracy = accuracy_score(y, prediction)
    precision = precision_score(y, prediction, zero_division=0)
    recall = recall_score(y, prediction)
    f1 = f1_score(y, prediction)
    f2 = fbeta_score(y, prediction, beta=2)
    return accuracy, precision, recall, f1, f2


def get_classification_report(y, prediction, target_names=None):
    report = classification_report(y, prediction, target_names=target_names)
    return report


def get_confusion_matrix(y, prediction):
    cm = confusion_matrix(y, prediction)
    return cm


def print_report(y, prediction, target_names=None):
    # print("\n=== CLASSIFICATION REPORT ===")
    # print(get_classification_report(y, prediction, target_names=target_names))

    accuracy, precision, recall, f1, f2 = get_score_classification(y, prediction)
    # print("\n=== PERFORMANCE DU MODÈLE ===")

    print_title("PERFORMANCE DU MODÈLE")
    print_col(f" Accuracy (Exactitude):    {accuracy:>8.3f} │ (TP+TN)/(TP+TN+FP+FN)")
    print_col(
        f" Precision (Précision):    {precision:>8.3f} │ TP/(TP+FP) | Minimiser les faux positifs. "
    )
    print_col(
        f" Recall (Sensibilité):     {recall:>8.3f} │ TP/(TP+FN) | Minimiser les faux négatif. "
    )
    print_col(
        f" F1-score:                 {f1:>8.3f} │ 2*Precision*Recall/(Precision+Recall)"
    )
    print_col(
        f" F2-score:                 {f2:>8.3f} │ 5*Precision*Recall/(4*Precision+Recall) | Privilégie le rappel)"
    )
    print_end()
    # "\n=== MATRICE DE CONFUSION ==="
    print_confusion_matrix_text(y, prediction, class_names=target_names)

    return accuracy, precision, recall, f1, f2


def print_confusion_matrix_text(y_true, y_pred, class_names=None):
    """Affichage textuel formaté de la matrice"""

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = [f"Classe {i}" for i in range(len(cm))]

    print_title("MATRICE DE CONFUSION")

    # En-tête
    print_col(f" {'RÉALITÉ \\ PRÉDICTION':<20}")
    print(f"| {'':>12}", end="")
    for name in class_names:
        print(f"{name:>12}", end="")
    print()

    # Lignes de la matrice
    for i, true_name in enumerate(class_names):
        print(f"| {true_name:>12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>12}", end="")
        print()

    # Pour classification binaire : détail TP, FP, etc.
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        print("|")
        print_col(f" Détail (classe positive = {class_names[1]}):")
        print_col(f" ✅ Vrais Positifs (TP):  {tp}")
        print_col(f" ❌ Faux Positifs (FP):   {fp}")
        print_col(f" ❌ Faux Négatifs (FN):   {fn}")
        print_col(f" ✅ Vrais Négatifs (TN):  {tn}")
        print_end()


def print_cross_validation_scores(model, X, y):
    """Affichage des scores de validation croisée"""
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f2": make_scorer(fbeta_score, beta=2),  # Directement dans le dict
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

    # Calcul CV pour chaque métrique
    cv_coefficients = {}
    for metric in ["accuracy", "precision", "recall", "f1", "f2"]:
        metric_scores = scores[f"test_{metric}"]
        mean_score = metric_scores.mean()
        std_score = metric_scores.std()
        cv_coef = std_score / mean_score if mean_score != 0 else float("inf")
        cv_coefficients[metric] = cv_coef

    accuracy_score = scores["test_accuracy"].mean()
    precision_score = scores["test_precision"].mean()
    recall_score = scores["test_recall"].mean()
    f1_score = scores["test_f1"].mean()
    f2_score = scores["test_f2"].mean()

    print_title(f"VALIDATION CROISÉE {type(model).__name__}")
    print_col(
        f" Accuracy: {accuracy_score:.3f} (+/- {scores['test_accuracy'].std() * 2:.3f}) [CV: {cv_coefficients['accuracy']:.3f}]"
    )
    print_col(
        f" Précision: {precision_score:.3f} (+/- {scores['test_precision'].std() * 2:.3f}) [CV: {cv_coefficients['precision']:.3f}]"
    )
    print_col(
        f" Rappel:    {recall_score:.3f} (+/- {scores['test_recall'].std() * 2:.3f}) [CV: {cv_coefficients['recall']:.3f}]"
    )
    print_col(
        f" F1-Score:  {f1_score:.3f} (+/- {scores['test_f1'].std() * 2:.3f}) [CV: {cv_coefficients['f1']:.3f}]"
    )
    print_col(
        f" F2-Score:  {f2_score:.3f} (+/- {scores['test_f2'].std() * 2:.3f}) [CV: {cv_coefficients['f2']:.3f}]"
    )
    print("|")
    stable = {}
    for metric in ["precision", "recall", "f1", "f2"]:
        stable[metric] = (
            "✅ Stable"
            if cv_coefficients[metric] < 0.05
            else "⚠️ Modéré" if cv_coefficients[metric] < 0.10 else "❌ Instable"
        )
        print_col(
            f" Stabilité {metric}:  {cv_coefficients[metric]:.3f} {stable[metric]}"
        )
    print_end()
    return (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        f2_score,
        str(stable),
    )


def find_optimal_threshold_pr(y_true, y_proba, metric="f1"):
    """Trouve le seuil optimal en utilisant la courbe PR"""

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if metric == "f1":
        # F1-Score pour chaque seuil
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)  # Remplace NaN par 0
        best_idx = np.argmax(f1_scores)
        best_score = f1_scores[best_idx]

    elif metric == "f2":
        # F2-Score pour chaque seuil
        f2_scores = 5 * (precision * recall) / (4 * precision + recall)
        f2_scores = np.nan_to_num(f2_scores)
        best_idx = np.argmax(f2_scores)
        best_score = f2_scores[best_idx]

    # Seuil optimal
    if best_idx < len(thresholds):
        optimal_threshold = thresholds[best_idx]
    else:
        optimal_threshold = 0.5

    # Visualisation
    plt.figure(figsize=(12, 4))

    # Courbe PR avec point optimal
    plt.subplot(1, 3, 1)
    plt.plot(recall, precision, "b-", linewidth=2)
    plt.plot(
        recall[best_idx],
        precision[best_idx],
        "ro",
        markersize=10,
        label=f"Optimal {metric.upper()}",
    )
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title("Courbe PR - Point Optimal")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Score vs Seuil
    plt.subplot(1, 3, 2)
    if metric == "f1":
        plt.plot(thresholds, f1_scores[:-1], "g-", linewidth=2)
        plt.ylabel("F1-Score")
    else:
        plt.plot(thresholds, f2_scores[:-1], "g-", linewidth=2)
        plt.ylabel("F2-Score")

    plt.axvline(optimal_threshold, color="red", linestyle="--")
    plt.xlabel("Seuil")
    plt.title(f"{metric.upper()}-Score vs Seuil")
    plt.grid(True, alpha=0.3)

    # Distribution des probabilités
    plt.subplot(1, 3, 3)
    plt.hist(y_proba[y_true == 0], bins=30, alpha=0.7, label="Classe 0")
    plt.hist(y_proba[y_true == 1], bins=30, alpha=0.7, label="Classe 1")
    plt.axvline(
        optimal_threshold,
        color="red",
        linestyle="--",
        label=f"Seuil optimal = {optimal_threshold:.3f}",
    )
    plt.xlabel("Probabilité")
    plt.ylabel("Fréquence")
    plt.title("Distribution des Scores")
    plt.legend()

    plt.tight_layout()
    plt.show()

    print_title("SEUIL OPTIMAL")
    print_col(f" Seuil optimal pour {metric.upper()}: {optimal_threshold:.3f}")
    print_col(f" Score optimal: {best_score:.3f}")
    print_col(f" Précision: {precision[best_idx]:.3f}")
    print_col(f" Rappel: {recall[best_idx]:.3f}")
    print_end()

    return optimal_threshold, best_score
