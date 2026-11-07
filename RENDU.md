# Rattrapage DevOps / MLOps – Dyhia BOUDJEMA

## Partie 3 – Choix du meilleur modèle

Je choisis le modèle entraîné sur 10 époques. D'après les Model metrics et la
comparaison des deux modèles dans MLflow, sa `val_accuracy` (la proportion
d'images bien classées parmi les images de validation, que le modèle n'a jamais
vues pendant l'entraînement) est plus haute : 0.808 contre 0.370. Sa `val_loss`
est aussi plus basse : 0.476 contre 1.014. Comme ses métriques de validation
sont meilleures que celles d'entraînement, il n'y a aucun signe de
sur-apprentissage. Au contraire, le modèle à 5 époques ne s'est pas encore
stabilisé : il fait à peine mieux que le hasard (`val_accuracy` = 0.370, alors
qu'en répondant au hasard parmi les trois formes on aurait raison une fois sur
trois).

Ce choix est aussi automatisé : `services/train/register_best.py` compare les
runs avec le même critère (`val_accuracy`, puis `val_loss`), enregistre le
meilleur modèle dans le Model Registry et le passe en Production, seulement
s'il est strictement meilleur que celui déjà en Production.

## Captures

Partie 2 – Pipeline DevOps
- `captures/partie2_01_pipeline_reussi.png` : le pipeline GitHub Actions réussi
- `captures/partie2_02_liste_executions.png` : les exécutions déclenchées par les push
- `captures/partie2_03_job_tests.png` : installation des dépendances et tests de l'API

Partie 3 – Pipeline MLOps
- `captures/partie3_01_runs_mlflow.png` : les deux exécutions dans MLflow
- `captures/partie3_02_entrainement_epochs-5_terminal.png` et `partie3_03_entrainement_epochs-10_terminal.png` : les entraînements
- `captures/partie3_04_metriques_epochs-5.png` et `partie3_05_metriques_epochs-10.png` : les courbes de chaque run
- `captures/partie3_06_comparaison.png` : la comparaison des deux runs (val_accuracy et val_loss)
- `captures/partie3_07_modele_enregistre.png` : le modèle enregistré en Production

Partie 4 – Monitoring
- `captures/partie4_01_metrics_api.png` : la route /metrics de l'API
- `captures/partie4_02_prometheus_target_up.png` : la cible de l'API collectée par Prometheus
- `captures/partie4_03_dashboard_grafana.png` : le dashboard Grafana pendant un envoi de requêtes

Partie 5 – Kubernetes
- `captures/partie5_01_kubectl_get.png` : pods, services, volume et jobs dans le namespace geometrie
- `captures/partie5_02_register.png` : enregistrement du meilleur modèle par le job register
- `captures/partie5_03_prediction_api.png` : l'API déployée sur Kubernetes répond
- `captures/partie5_04_grafana.png` : le dashboard Grafana déployé sur Kubernetes
