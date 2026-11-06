import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow

AUTOTUNE = tf.data.experimental.AUTOTUNE

def make_datasets(data_dir, img_height=28, img_width=28, batch_size=32):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names

def build_model(num_classes, learning_rate):
    model = keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Flatten(input_shape=(28,28,3)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    data_dir = os.environ.get("DATA_DIR", "/data/knowledge")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "geometrie")

    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    epochs = int(os.environ.get("EPOCHS", "30"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.01"))

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(exp_name)

    # Même graine que le découpage train/validation : les runs partent des mêmes poids,
    # seul le nombre d'époques les différencie
    tf.keras.utils.set_random_seed(123)

    train_ds, val_ds, class_names = make_datasets(data_dir, batch_size=batch_size)
    model = build_model(num_classes=len(class_names), learning_rate=learning_rate)

    with mlflow.start_run(run_name=f"epochs-{epochs}"):
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("classes", ",".join(class_names))

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        # log métriques à chaque époque (courbes visibles dans MLflow)
        for step in range(epochs):
            mlflow.log_metrics({
                "accuracy": float(history.history["accuracy"][step]),
                "loss": float(history.history["loss"][step]),
                "val_accuracy": float(history.history["val_accuracy"][step]),
                "val_loss": float(history.history["val_loss"][step]),
            }, step=step)

        # log modèle : enregistré dans le Model Registry seulement après comparaison des runs
        mlflow.tensorflow.log_model(model, artifact_path="model")

        print(f"Run epochs-{epochs} terminé, modèle loggé dans MLflow")

if __name__ == "__main__":
    main()
