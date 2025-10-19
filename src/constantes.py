import os

NOTEBOOK_NAME = "03-Modeling"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5010")
TAGS_DEFAULT = {
    "project": "lung_cancer",
    "team": "DataScience",
    "department": "Openclassrooms",
    "status": "active",
    "version": "1.0",
    "owner": "francois.hellebuyck",
    "data_version": "v1.0",
    "dataset": "survey_lung_cancer_features.parquet",
    "mlflow.user": "francois.hellebuyck",
    "mlflow.source.name": NOTEBOOK_NAME,
    "mlflow.source.type": "notebook",
}
TAGS_BASELINE = {"experiment_type": "baseline_comparison", "model_type": "baseline"}
TAGS_CV = {"experiment_type": "cross_validation", "model_type": "cross_validation"}
TAGS_TO = {
    "experiment_type": "threshold_optimization",
    "model_type": "threshold_optimization",
}
TAGS_LGBMO = {
    "experiment_type": "LGBM_optimization",
    "model_type": "LGBM_optimization",
}

TAGS_FINAL_MODEL = {"experiment_type": "final_model", "model_type": "LGBM_Final_model"}

EXPERIMENT = "Lung Cancer Detection"
