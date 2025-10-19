import mlflow
from mlflow.tracking import MlflowClient
from src.constantes import TRACKING_URI
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import os
import shap
import pickle


def create_experiment_with_metadata(experiment_name, description, tags=None):
    """
    Crée ou récupère une expérience avec description et tags

    Args:
        experiment_name: nom de l'expérience
        description: description détaillée
        tags: dictionnaire de tags (optionnel)

    Returns:
        experiment_id
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Vérifier si l'expérience existe
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Créer la nouvelle expérience
        experiment_id = client.create_experiment(experiment_name)
        print(f"✅ Expérience créée: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"ℹ️  Expérience existante: {experiment_name}")

    # Ajouter la description
    if description:
        client.set_experiment_tag(experiment_id, "mlflow.note.content", description)
        print("✅ Description ajoutée")

    # Ajouter les tags
    if tags:
        for key, value in tags.items():
            client.set_experiment_tag(experiment_id, key, value)
        print(f"✅ {len(tags)} tag(s) ajouté(s)")

    return experiment_id


def start_experiment(experiment_name, description, tags=None):
    """
    Démarre une expérience MLflow avec description et tags

    Args:
        experiment_name: nom de l'expérience
        description: description détaillée
        tags: dictionnaire de tags (optionnel)

    Returns:
        experiment_id
    """
    experiment_id = create_experiment_with_metadata(experiment_name, description, tags)
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_global_params_and_tags(X_train, X_test, df, tags):
    """
    Logue les paramètres globaux et les tags pour une expérience MLflow
    Args:
        X_train: données d'entraînement
        X_test: données de test
        df: DataFrame complète (pour loguer la source des données)
        name: nom du modèle
        tags: dictionnaire de tags à ajouter
    """
    # Loguer les paramètres globaux (une seule fois)
    mlflow.log_param("data_version", "v1.0")
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("features", X_train.shape[1])

    # Plusieurs tags à la fois
    mlflow.set_tags(tags)

    dataset = mlflow.data.from_pandas(
        df,
        source="data/processed/survey_lung_cancer_features.parquet",
        name="survey_lung_cancer_features",
        targets="LUNG_CANCER",
    )
    mlflow.log_input(dataset, context="training")


def log_base_metrics(accuracy, precision, recall, f1, f2):
    """
    Logue les métriques de base et affiche le rapport de classification

    Args:
        accuracy: précision globale
        precision: précision
        recall: rappel
        f1: score F1
        f2: score F2
    """
    # Loguer les métriques principales
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1)
    mlflow.log_metric("f2-score", f2)


def log_complete_system_metrics():
    """Logger toutes les métriques système pertinentes"""

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
    cpu_count = psutil.cpu_count()

    # Mémoire
    memory = psutil.virtual_memory()

    # Disque
    disk = psutil.disk_usage("/")

    # Process actuel
    process = psutil.Process()
    process_memory = process.memory_info()

    metrics = {
        # CPU
        "system_cpu_percent": cpu_percent,
        "system_cpu_count": cpu_count,
        # Mémoire système
        "system_memory_total_gb": memory.total / (1024**3),
        "system_memory_available_gb": memory.available / (1024**3),
        "system_memory_used_percent": memory.percent,
        # Disque
        "system_disk_used_percent": disk.percent,
        "system_disk_free_gb": disk.free / (1024**3),
        # Process
        "process_memory_rss_mb": process_memory.rss / (1024**2),
        "process_memory_vms_mb": process_memory.vms / (1024**2),
    }

    return metrics


def log_feature_importance_with_names(pipeline, X_train, mlflow_enabled=True):
    """
    Logger les feature importances avec les vrais noms des features

    Args:
        pipeline: Pipeline sklearn/imblearn
        X_train: Données d'entraînement (DataFrame ou array)
        mlflow_enabled: Logger dans MLflow ou non
    """

    # Extraire le modèle LightGBM
    if hasattr(pipeline, "named_steps"):
        lgb_model = pipeline.named_steps.get("classifier") or pipeline.named_steps.get(
            "model"
        )
    else:
        lgb_model = pipeline

    # Récupérer les noms des features ORIGINAUX
    if hasattr(X_train, "columns"):
        feature_names = X_train.columns.tolist()
    elif hasattr(lgb_model, "feature_name_"):
        # Essayer de récupérer depuis le modèle
        feature_names = lgb_model.feature_name_
    else:
        # Fallback : noms génériques
        n_features = len(lgb_model.feature_importances_)
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Récupérer les importances
    feature_importances = lgb_model.feature_importances_

    # Vérifier que les longueurs correspondent
    if len(feature_names) != len(feature_importances):
        print(f"⚠️  Nombre de features: {len(feature_names)}")
        print(f"⚠️  Nombre d'importances: {len(feature_importances)}")
        # Ajuster si nécessaire
        if len(feature_names) > len(feature_importances):
            feature_names = feature_names[: len(feature_importances)]
        else:
            feature_names = feature_names + [
                f"feature_{i}"
                for i in range(len(feature_names), len(feature_importances))
            ]

    # Créer le DataFrame
    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    ).sort_values("importance", ascending=False)

    fi_df["rank"] = range(1, len(fi_df) + 1)

    # Afficher
    print("\n🎯 Top 20 Features:")
    print(fi_df.head(20).to_string(index=False))

    # === GRAPHIQUE ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Graphique 1 : Top 20
    top_20 = fi_df.head(20)
    axes[0].barh(range(len(top_20)), top_20["importance"], color="steelblue")
    axes[0].set_yticks(range(len(top_20)))
    axes[0].set_yticklabels(top_20["feature"], fontsize=10)
    axes[0].set_xlabel("Importance", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Feature", fontsize=12, fontweight="bold")
    axes[0].set_title("Top 20 Feature Importances", fontsize=14, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    # Graphique 2 : Distribution des importances
    axes[1].hist(
        fi_df["importance"], bins=30, color="coral", edgecolor="black", alpha=0.7
    )
    axes[1].set_xlabel("Importance Value", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Number of Features", fontsize=12, fontweight="bold")
    axes[1].set_title(
        "Distribution of Feature Importances", fontsize=14, fontweight="bold"
    )
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("feature_importance_complete.png", dpi=100, bbox_inches="tight")

    if mlflow_enabled:
        mlflow.log_artifact("feature_importance_complete.png")

    plt.close()

    if mlflow_enabled:
        os.remove("feature_importance_complete.png")

    # === CSV ===
    fi_df.to_csv("feature_importance.csv", index=False)

    if mlflow_enabled:
        mlflow.log_artifact("feature_importance.csv")
        os.remove("feature_importance.csv")

    # === LOGGER TOP 10 DANS MLFLOW ===
    if mlflow_enabled:
        for i, row in enumerate(fi_df.head(10).itertuples(), 1):
            mlflow.log_metric(f"top_{i}_importance", float(row.importance))
            mlflow.log_param(f"top_{i}_feature", row.feature)

    print("✅ Feature importances sauvegardées avec les vrais noms!")

    return fi_df


def log_shap_artifacts(
    pipeline, X_test, classifier_name="classifier", scaler_name="scaler", prefix="shap"
):
    """
    Calcule et sauvegarde les valeurs SHAP et graphiques dans MLflow

    Parameters:
    -----------
    pipeline : Pipeline
        Pipeline imblearn/sklearn entraîné
    X_test : DataFrame ou array
        Données de test
    classifier_name : str, default='classifier'
        Nom de l'étape du classifier dans le pipeline
    scaler_name : str, default='scaler'
        Nom de l'étape du scaler dans le pipeline
    prefix : str, default='shap'
        Préfixe pour les noms de fichiers

    Returns:
    --------
    shap_values : array ou list
        Valeurs SHAP calculées
    explainer : shap.TreeExplainer
        L'explainer SHAP
    """

    # 1. Extraire le classifier du pipeline
    model = pipeline.named_steps[classifier_name]

    # 2. Transformer X_test avec le scaler
    X_test_transformed = pipeline.named_steps[scaler_name].transform(X_test)

    # 3. Calculer les valeurs SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    # 4. Gérer binaire vs multiclasse
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]  # Classe positive pour binaire
    else:
        shap_values_plot = shap_values

    # 5. Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_plot,
        X_test_transformed,
        feature_names=X_test.columns if hasattr(X_test, "columns") else None,
        show=False,
    )
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    filename = f"{prefix}_summary_plot.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    mlflow.log_artifact(filename)
    plt.show()
    plt.close()
    os.remove(filename)

    # 6. Bar plot (importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_plot,
        X_test_transformed,
        feature_names=X_test.columns if hasattr(X_test, "columns") else None,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    filename = f"{prefix}_importance_bar.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    mlflow.log_artifact(filename)
    plt.show()
    plt.close()
    os.remove(filename)

    # 7. Sauvegarder les valeurs SHAP brutes
    filename = f"{prefix}_values.pkl"
    with open(filename, "wb") as f:
        pickle.dump(shap_values, f)
    mlflow.log_artifact(filename)
    os.remove(filename)

    # 8. Sauvegarder l'explainer
    filename = f"{prefix}_explainer.pkl"
    with open(filename, "wb") as f:
        pickle.dump(explainer, f)
    mlflow.log_artifact(filename)
    os.remove(filename)

    print("✅ Tous les artifacts SHAP sauvegardés dans MLflow!")

    return shap_values, explainer
