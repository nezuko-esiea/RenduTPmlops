# Reconnaissance de formes géométriques – DevOps / MLOps

Dépôt : https://github.com/nezuko-esiea/RenduTPmlops

Une IA TensorFlow reconnaît des cercles, rectangles et triangles. Le projet
contient l'entraînement suivi par MLflow, une API FastAPI de prédiction, le
monitoring Prometheus / Grafana, un pipeline GitHub Actions et les manifestes
Kubernetes.

Les images d'entraînement (`geometrie/knowledge`) et de test
(`geometrie/sample_images`) viennent de l'archive `geometrie.zip` du module.

## Prérequis

- Docker Desktop
- Pour Kubernetes : minikube et kubectl

## Lancer le projet avec Docker Compose

```bash
docker compose up -d --build
docker compose run --rm --build -e EPOCHS=5 train
docker compose run --rm -e EPOCHS=10 train
docker compose run --rm register
```

1. Démarre MLflow, l'API, Prometheus et Grafana.
2. et 3. Entraînent le modèle sur 5 puis 10 époques ; chaque run est enregistré dans MLflow.
4. Compare les runs, enregistre le meilleur modèle et le passe en Production.

L'API charge le modèle en Production à la première prédiction :

```bash
curl http://localhost:8000/health
curl -F "file=@geometrie/sample_images/test1.png" http://localhost:8000/predict
```

Sous PowerShell, utiliser `curl.exe`. La documentation interactive de l'API est
sur http://localhost:8000/docs.

Interfaces :
- MLflow : http://localhost:5000
- Prometheus : http://localhost:9090
- Grafana : http://localhost:3000 (admin / admin), dashboard MLOps > API Géométrie - Monitoring

Pour générer du trafic sur l'API : `bash load.sh`.

Entre conteneurs, les services se joignent par leur nom (`http://mlflow:5000`,
`api:8000`) ; depuis le navigateur, on passe par `localhost`.

## Pipeline CI

`.github/workflows/ci.yml` se lance à chaque push : vérification de la syntaxe
(compileall et ruff), tests de l'API (`tests/test_api.py`), build de l'image
Docker de l'API, puis résultat du pipeline.

## Déployer sur Kubernetes (minikube)

Les manifestes de `k8s/` ont été générés avec `kompose convert` puis corrigés
(voir l'historique git).

```bash
docker compose --profile train build
minikube start
minikube image load rattrapagedevops-api:latest
minikube image load rattrapagedevops-train:latest
```

Dans un autre terminal, à laisser ouvert pendant l'entraînement :

```bash
minikube mount "<chemin du projet>/geometrie/knowledge:/data/knowledge"
```

Puis :

```bash
kubectl apply -k k8s
kubectl apply -f k8s/jobs/train-epochs-5.yaml
kubectl wait --for=condition=complete job/train-epochs-5 -n geometrie --timeout=10m
kubectl apply -f k8s/jobs/train-epochs-10.yaml
kubectl wait --for=condition=complete job/train-epochs-10 -n geometrie --timeout=10m
kubectl apply -f k8s/jobs/register.yaml
kubectl logs job/register -n geometrie
```

Accès à l'API et à Grafana (chaque commande dans un terminal à laisser ouvert) :

```bash
kubectl port-forward -n geometrie svc/api 18000:8000
kubectl port-forward -n geometrie svc/grafana 13000:3000
```

L'API répond alors sur http://localhost:18000 et Grafana sur http://localhost:13000.

## Rendu

- `RENDU.md` : justification du choix du modèle et liste des captures
- `captures/` : captures d'écran de chaque partie
