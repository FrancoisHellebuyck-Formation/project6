docker run -p 5010:5000 ghcr.io/mlflow/mlflow:v3.4.0  mlflow server --host 0.0.0.0

# Lancer les services
docker-compose up -d

# Vérifier que les services sont en cours d'exécution
docker-compose ps

# Voir les logs
docker-compose logs -f mlflow

Accès

MLflow UI : http://localhost:5000
PostgreSQL : localhost:5432

Arrêt et nettoyage

# Arrêter les services
docker-compose down

# Arrêter et supprimer les volumes (attention : perte de données)
docker-compose down -v


Configuration
Le fichier utilise :

PostgreSQL 15 comme base de données backend
Volumes persistants pour les données PostgreSQL et les artefacts MLflow
Healthcheck pour s'assurer que PostgreSQL est prêt avant de démarrer MLflow
Network isolé pour la communication entre les services

Les identifiants par défaut sont :

User : mlflow
Password : mlflow
Database : mlflowdb

N'oubliez pas de modifier ces identifiants pour un environnement de production !


# Build et lancement
docker-compose up -d --build

# Voir les logs
docker-compose logs -f mlflow

# Rebuild uniquement le service mlflow
docker-compose up -d --build mlflow