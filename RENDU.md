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

### Captures

- `captures/partie3_01_runs_mlflow.png` : les deux exécutions dans MLflow
- `captures/partie3_02_entrainement_epochs-5_terminal.png` et `partie3_03_entrainement_epochs-10_terminal.png` : les entraînements
- `captures/partie3_04_metriques_epochs-5.png` et `partie3_05_metriques_epochs-10.png` : les courbes de chaque run
- `captures/partie3_06_comparaison.png` : la comparaison des deux runs (val_accuracy et val_loss)
- `captures/partie3_07_modele_enregistre.png` : le modèle enregistré en Production
