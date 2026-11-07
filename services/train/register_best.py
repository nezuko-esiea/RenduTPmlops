import os
import mlflow
from mlflow.tracking import MlflowClient

def main():
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "geometrie")
    model_name = os.environ.get("MODEL_NAME", "geometrie-model")

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        raise SystemExit(f"Aucune expérience '{exp_name}' : lancer d'abord l'entraînement")

    # meilleur run = val_accuracy la plus haute, puis val_loss la plus basse
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.val_accuracy DESC", "metrics.val_loss ASC"],
    )
    if not runs:
        raise SystemExit("Aucun run terminé à comparer")

    print("Comparaison des runs (validation) :")
    for run in runs:
        metrics = run.data.metrics
        print(f"  {run.info.run_name}: val_accuracy={metrics.get('val_accuracy', 0):.3f} "
              f"val_loss={metrics.get('val_loss', 0):.3f}")

    best = runs[0]
    print(f"Meilleur run : {best.info.run_name}")

    # le modèle en Production n'est remplacé que par un run strictement meilleur
    versions = client.search_model_versions(f"name='{model_name}'")
    production = next((v for v in versions if v.current_stage == "Production"), None)
    if production is not None:
        prod_metrics = client.get_run(production.run_id).data.metrics
        best_metrics = best.data.metrics
        best_score = (best_metrics["val_accuracy"], -best_metrics["val_loss"])
        prod_score = (prod_metrics["val_accuracy"], -prod_metrics["val_loss"])
        if best_score <= prod_score:
            print(f"{model_name} version {production.version} (en Production) est déjà "
                  "au moins aussi bon : aucun changement")
            return

    # on réutilise la version si ce run est déjà enregistré
    version = next((v for v in versions if v.run_id == best.info.run_id), None)
    if version is None:
        version = mlflow.register_model(f"runs:/{best.info.run_id}/model", model_name)
        print(f"Enregistré : {model_name} version {version.version}")

    if version.current_stage != "Production":
        client.transition_model_version_stage(
            model_name, version.version, "Production", archive_existing_versions=True
        )
    print(f"{model_name} version {version.version} est en Production")

if __name__ == "__main__":
    main()
