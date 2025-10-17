import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from src.visualization.visu_text import print_title, print_col, print_end

# ================================
# 1. OPTIMISATION DU SEUIL POUR F2
# ================================


def optimize_threshold_f2(model, X_val, y_val, thresholds=None):
    """
    Trouve le seuil optimal pour maximiser le F2-score

    Parameters:
    - model: modèle entraîné
    - X_val, y_val: données de validation
    - thresholds: seuils à tester (par défaut: 0.1 à 0.9)

    Returns:
    - best_threshold: seuil optimal
    - best_f2: meilleur F2-score
    - results_df: DataFrame avec tous les résultats
    """

    # Prédictions probabilistes
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Seuils à tester (plus dense autour de 0.3-0.6 pour le déséquilibre)
    if thresholds is None:
        thresholds = np.concatenate(
            [
                np.arange(0.1, 0.3, 0.05),  # Seuils bas
                np.arange(0.3, 0.6, 0.01),  # Zone critique (plus dense)
                np.arange(0.6, 0.9, 0.05),  # Seuils hauts
            ]
        )

    results = []

    # print("🎯 Optimisation du seuil pour F2-score")
    # print("=" * 45)

    for threshold in thresholds:
        # Prédictions binaires avec ce seuil
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calcul des métriques
        try:
            f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)

            results.append(
                {
                    "threshold": threshold,
                    "f2_score": f2,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "predicted_positive": y_pred.sum(),
                    "predicted_negative": (y_pred == 0).sum(),
                }
            )
        except ZeroDivisionError:
            # En cas d'erreur (division par zéro, etc.)
            results.append(
                {
                    "threshold": threshold,
                    "f2_score": 0,
                    "f1_score": 0,
                    "precision": 0,
                    "recall": 0,
                    "predicted_positive": 0,
                    "predicted_negative": len(y_pred),
                }
            )

    # Conversion en DataFrame
    results_df = pd.DataFrame(results)

    # Trouver le meilleur seuil
    best_idx = results_df["f2_score"].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]
    best_f2 = results_df.loc[best_idx, "f2_score"]

    # print(f"🏆 SEUIL OPTIMAL:")
    # print(f"Seuil: {best_threshold:.3f}")
    # print(f"F2-Score: {best_f2:.4f}")
    # print(f"F1-Score: {results_df.loc[best_idx, 'f1_score']:.4f}")
    # print(f"Précision: {results_df.loc[best_idx, 'precision']:.4f}")
    # print(f"Rappel: {results_df.loc[best_idx, 'recall']:.4f}")

    return best_threshold, best_f2, results_df


# ================================
# 2. VISUALISATION DE L'OPTIMISATION
# ================================


def plot_threshold_optimization(results_df, best_threshold):
    """Visualise l'optimisation du seuil"""

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    fig.suptitle(
        "Optimisation du Seuil de Classification", fontsize=16, fontweight="bold"
    )

    # 1. Courbe F2-score vs seuil
    ax1 = axes[0]
    ax1.plot(
        results_df["threshold"],
        results_df["f2_score"],
        "b-",
        linewidth=2,
        label="F2-Score",
    )
    ax1.plot(
        results_df["threshold"],
        results_df["f1_score"],
        "g--",
        linewidth=2,
        label="F1-Score",
    )
    ax1.axvline(
        x=best_threshold,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"Optimal: {best_threshold:.3f}",
    )
    ax1.set_xlabel("Seuil")
    ax1.set_ylabel("Score")
    ax1.set_title("F2 vs F1 Score par Seuil")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Précision vs Rappel
    ax2 = axes[1]
    ax2.plot(
        results_df["threshold"],
        results_df["precision"],
        "r-",
        linewidth=2,
        label="Précision",
    )
    ax2.plot(
        results_df["threshold"], results_df["recall"], "b-", linewidth=2, label="Rappel"
    )
    ax2.axvline(
        x=best_threshold,
        color="red",
        linestyle=":",
        alpha=0.7,
        label=f"Optimal: {best_threshold:.3f}",
    )
    ax2.set_xlabel("Seuil")
    ax2.set_ylabel("Score")
    ax2.set_title("Précision vs Rappel")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # # 3. Nombre de prédictions positives
    # ax3 = axes[1, 0]
    # ax3.plot(results_df['threshold'], results_df['predicted_positive'], 'purple', linewidth=2)
    # ax3.axvline(x=best_threshold, color='red', linestyle=':', alpha=0.7, label=f'Optimal: {best_threshold:.3f}')
    # ax3.set_xlabel('Seuil')
    # ax3.set_ylabel('Nombre de prédictions positives')
    # ax3.set_title('Impact du Seuil sur les Prédictions')
    # ax3.legend()
    # ax3.grid(True, alpha=0.3)

    # # 4. Heat map des métriques par seuil
    # ax4 = axes[1, 1]

    # # Créer une matrice pour le heatmap
    # metrics_for_heatmap = results_df[['f2_score', 'f1_score', 'precision', 'recall']].T

    # # Sélectionner quelques seuils représentatifs pour l'affichage
    # step = max(1, len(results_df) // 20)
    # selected_indices = range(0, len(results_df), step)
    # selected_thresholds = results_df.iloc[selected_indices]['threshold'].round(2)

    # im = ax4.imshow(metrics_for_heatmap.iloc[:, selected_indices], cmap='RdYlGn', aspect='auto')
    # ax4.set_xticks(range(len(selected_indices)))
    # ax4.set_xticklabels(selected_thresholds, rotation=45)
    # ax4.set_yticks(range(4))
    # ax4.set_yticklabels(['F2', 'F1', 'Precision', 'Recall'])
    # ax4.set_title('Heatmap des Métriques')

    # Colorbar
    # plt.colorbar(im, ax=ax4, shrink=0.8)

    plt.tight_layout()
    plt.show()


# ================================
# 3. OPTIMISATION AVANCÉE AVEC COURBE PR
# ================================


def advanced_threshold_optimization(model, X_val, y_val):
    """Optimisation avancée avec courbe Precision-Recall"""

    # Prédictions probabilistes
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Courbe Precision-Recall
    precisions, recalls, thresholds_pr = precision_recall_curve(y_val, y_pred_proba)

    # Calculer F2 pour chaque point de la courbe PR
    f2_scores = []
    for p, r in zip(precisions, recalls):
        if p + r > 0:
            f2 = (1 + 2**2) * (p * r) / ((2**2 * p) + r)
        else:
            f2 = 0
        f2_scores.append(f2)

    f2_scores = np.array(f2_scores)

    # Trouver le meilleur seuil
    best_idx = np.argmax(f2_scores)

    # Attention: thresholds_pr a une longueur différente de precisions/recalls
    if best_idx < len(thresholds_pr):
        best_threshold_pr = thresholds_pr[best_idx]
    else:
        best_threshold_pr = thresholds_pr[-1]

    best_f2_pr = f2_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    print_title("OPTIMISATION F2 VIA COURBE PR:")
    print_col(f"Seuil optimal: {best_threshold_pr:.3f}")
    print_col(f"F2-Score: {best_f2_pr:.4f}")
    print_col(f"Précision: {best_precision:.4f}")
    print_col(f"Rappel: {best_recall:.4f}")

    # Visualisation courbe PR avec point optimal
    plt.figure(figsize=(4, 3))
    plt.plot(recalls, precisions, "b-", linewidth=2, label="Courbe Precision-Recall")
    plt.plot(
        best_recall,
        best_precision,
        "ro",
        markersize=10,
        label=f"Optimal F2 (seuil={best_threshold_pr:.3f})",
    )
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title("Courbe Precision-Recall avec Seuil Optimal F2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return best_threshold_pr, best_f2_pr


# ================================
# 4. ÉVALUATION AVEC SEUIL OPTIMAL
# ================================


def evaluate_with_optimal_threshold(model, X_val, y_val, optimal_threshold):
    """Évaluation complète avec le seuil optimal"""

    # Prédictions avec seuil optimal
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

    # Prédictions avec seuil par défaut (0.5)
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

    # print("📊 COMPARAISON SEUIL OPTIMAL vs DÉFAUT")
    # print("=" * 50)

    # Métriques pour seuil optimal
    f2_optimal = fbeta_score(y_val, y_pred_optimal, beta=2)
    f1_optimal = f1_score(y_val, y_pred_optimal)
    precision_optimal = precision_score(y_val, y_pred_optimal)
    recall_optimal = recall_score(y_val, y_pred_optimal)

    # Métriques pour seuil par défaut
    f2_default = fbeta_score(y_val, y_pred_default, beta=2)
    f1_default = f1_score(y_val, y_pred_default)
    precision_default = precision_score(y_val, y_pred_default)
    recall_default = recall_score(y_val, y_pred_default)

    # Affichage comparatif
    # comparison_df = pd.DataFrame(
    #     {
    #         f"Seuil Optimal ({optimal_threshold:.3f})": [
    #             f2_optimal,
    #             f1_optimal,
    #             precision_optimal,
    #             recall_optimal,
    #         ],
    #         "Seuil Défaut (0.5)": [
    #             f2_default,
    #             f1_default,
    #             precision_default,
    #             recall_default,
    #         ],
    #         "Amélioration": [
    #             f2_optimal - f2_default,
    #             f1_optimal - f1_default,
    #             precision_optimal - precision_default,
    #             recall_optimal - recall_default,
    #         ],
    #     },
    #     index=["F2-Score", "F1-Score", "Précision", "Rappel"],
    # )

    # print(comparison_df.round(4))

    # # Matrices de confusion
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # # Matrice confusion seuil optimal
    # cm_optimal = confusion_matrix(y_val, y_pred_optimal)
    # sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    # axes[0].set_title(f'Matrice de Confusion - Seuil Optimal ({optimal_threshold:.3f})')
    # axes[0].set_xlabel('Prédiction')
    # axes[0].set_ylabel('Réalité')

    # # Matrice confusion seuil défaut
    # cm_default = confusion_matrix(y_val, y_pred_default)
    # sns.heatmap(cm_default, annot=True, fmt='d', cmap='Reds', ax=axes[1])
    # axes[1].set_title('Matrice de Confusion - Seuil Défaut (0.5)')
    # axes[1].set_xlabel('Prédiction')
    # axes[1].set_ylabel('Réalité')

    # plt.tight_layout()
    # plt.show()

    return {
        "optimal_threshold": optimal_threshold,
        "metrics_optimal": {
            "f2": f2_optimal,
            "f1": f1_optimal,
            "precision": precision_optimal,
            "recall": recall_optimal,
        },
        "metrics_default": {
            "f2": f2_default,
            "f1": f1_default,
            "precision": precision_default,
            "recall": recall_default,
        },
        "improvement": {"f2": f2_optimal - f2_default, "f1": f1_optimal - f1_default},
    }


# ================================
# 5. PIPELINE COMPLET D'OPTIMISATION
# ================================


def complete_threshold_optimization(model, X_val, y_val, plot_results=True):
    """Pipeline complet d'optimisation du seuil"""

    # print("🚀 OPTIMISATION COMPLÈTE DU SEUIL")
    # print("=" * 50)

    # 1. Optimisation classique
    best_threshold, best_f2, results_df = optimize_threshold_f2(model, X_val, y_val)

    # 2. Optimisation via courbe PR
    best_threshold_pr, best_f2_pr = advanced_threshold_optimization(model, X_val, y_val)

    # 3. Choisir le meilleur des deux
    if best_f2_pr > best_f2:
        final_threshold = best_threshold_pr
        final_f2 = best_f2_pr
        print_col(
            f"✅ Meilleur seuil trouvé via courbe PR: {final_threshold:.3f} (F2: {final_f2:.4f})"
        )
    else:
        final_threshold = best_threshold
        final_f2 = best_f2
        print_col(
            f"✅ Meilleur seuil trouvé via recherche systématique: {final_threshold:.3f} (F2: {final_f2:.4f})"
        )

    # 4. Visualisations
    if plot_results:
        plot_threshold_optimization(results_df, final_threshold)

    # 5. Évaluation finale
    evaluation_results = evaluate_with_optimal_threshold(
        model, X_val, y_val, final_threshold
    )

    print_col(" RÉSUMÉ FINAL:")
    print_col(f" Seuil optimal: {final_threshold:.3f}")
    print_col(f" Amélioration F2: +{evaluation_results['improvement']['f2']:.4f}")
    print_col(f" Amélioration F1: +{evaluation_results['improvement']['f1']:.4f}")

    return final_threshold, evaluation_results


# ================================
# 6. EXEMPLE D'UTILISATION
# ================================

"""
# UTILISATION DANS VOTRE PIPELINE:

# 1. Entraîner votre modèle
model = XGBClassifier(...)
model.fit(X_train_resampled, y_train_resampled)

# 2. Optimiser le seuil sur validation
optimal_threshold, eval_results = complete_threshold_optimization(model, X_val, y_val)

# 3. Évaluation finale sur test avec seuil optimal
y_test_pred_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_pred_proba >= optimal_threshold).astype(int)

final_f2 = fbeta_score(y_test, y_test_pred, beta=2)
print(f"F2-Score final sur test: {final_f2:.4f}")
"""
