from google.cloud import storage
import logging

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
)
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam

from tensorboard.plugins.hparams import api as hp


from MRIsegmentation.data import get_data_from_drive, holdout
from MRIsegmentation.params import (
    BUCKET_NAME,
    EXPERIMENT_NAME,
    MLFLOW_URI,
    GDRIVE_DATA_PATH,
)
from MRIsegmentation.model import get_model
from MRIsegmentation.mlflow import MLFlowBase
from MRIsegmentation.utils import (
    focal_tversky,
    tversky,
    tversky_loss,
    process_path,
    augment_data,
    flatten_mask,
    normalize,
)


def save_model_(model: Model, model_name: str):
    print(model.summary())
    model.save(
        f"{GDRIVE_DATA_PATH}{model_name}_final.h5",
    )

    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)
    # blob = bucket.blob("models/" + f"best_{model_name}.h5")
    # blob.upload_from_filename(f"best_{model_name}.h5")


def load_model_(model_name):
    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)
    # blob = bucket.blob("models/" + f"best_{model_name}.h5")
    # blob.download_to_filename(f"best_{model_name}.h5")

    # model = get_model()
    # model.load_weights(f"{GDRIVE_DATA_PATH}{model_name}_ckpt.tf")

    # model = tf.saved_model.load(f"{GDRIVE_DATA_PATH}{model_name}_ckpt")

    model = load_model(
        f"{GDRIVE_DATA_PATH}{model_name}_final.h5",
        custom_objects={
            "focal_tversky": focal_tversky,
            "tversky": tversky,
            "tversky_loss": tversky_loss,
        },
    )
    print(model.summary())

    return model


class Trainer(MLFlowBase):
    def __init__(self):
        super().__init__(EXPERIMENT_NAME, MLFLOW_URI)
        logging.basicConfig(level=logging.INFO)
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self.model = None

    def train(self, params):
        # iterate on models
        for model_name, model_params in params.items():

            hyper_params = model_params["hyper_params"]

            # create a mlflow training
            self.mlflow_create_run()

            # log params
            self.mlflow_log_param("model_name", model_name)
            for key, value in hyper_params.items():
                self.mlflow_log_param(key, value)

            # get data frm drive
            df = get_data_from_drive()
            logging.info(f"Data loaded: {df.shape}")

            # holdout
            self.ds_train, self.ds_val, self.ds_test = holdout(df)

            # log params
            logging.info(f"Loading model: {model_name}")

            # create model
            model = get_model(model_name)

            # compling model and callbacks functions
            adam = Adam(learning_rate=hyper_params["learning_rate"], epsilon=0.1)

            model.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])

            # callbacks
            # earlystopping = EarlyStopping(
            #    monitor="val_loss", mode="min", verbose=1, patience=30
            # )

            # save the best model with lower validation loss
            checkpointer = ModelCheckpoint(
                filepath=f"{GDRIVE_DATA_PATH}{model_name}_ckpt.h5",
                verbose=1,
                save_best_only=True,
                # save_weights_only=True,
            )

            # reduce learning rate when on a plateau
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                verbose=1,
                patience=10,
                min_delta=0.0001,
                factor=0.2,
            )

            # Tensorboard
            tensorboard = TensorBoard("logs/hparam_tuning")

            # Hyper parameters logging
            hyper_p = hp.KerasCallback("logs/hparam_tuning", hyper_params)

            batch_size = hyper_params["batch_size"]

            cardinality = df.shape[0]

            ds_train = (
                self.ds_train.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
                .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
                .map(flatten_mask, num_parallel_calls=tf.data.AUTOTUNE)
                .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(cardinality)
                .batch(batch_size=batch_size)
                .prefetch(2)
            )

            ds_val = (
                self.ds_val.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
                .map(flatten_mask, num_parallel_calls=tf.data.AUTOTUNE)
                .map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size=batch_size)
            )

            history = model.fit(
                ds_train,
                epochs=hyper_params["epochs"],
                callbacks=[
                    checkpointer,
                    # earlystopping,
                    reduce_lr,
                    tensorboard,
                    hyper_p,
                ],
                validation_data=ds_val,
            )

            save_model_(model, model_name)
            logging.info(f"Saving model: {model_name}")

            self.model = model

            # push best params & score to mlflow
            # for k, v in grid_search.best_params_.items():
            #    self.mlflow_log_param('best__' + k, v)

            # push metrics to mlflow
            self.mlflow_log_metric("loss", history.history["loss"][-1])
            self.mlflow_log_metric("val_loss", history.history["val_loss"][-1])
            self.mlflow_log_metric("tversky", history.history["tversky"][-1])
            self.mlflow_log_metric("val_tversky", history.history["val_tversky"][-1])

            # return the gridsearch in order to identify the best estimators and params

        return (f"goto {MLFLOW_URI}/#/experiments/{self.mlflow_experiment_id}", history)

    def evaluate(self, model_name="vgg19"):
        model: Model = load_model_(model_name)
        print(
            model.evaluate(
                self.ds_test.map(process_path)
                .map(flatten_mask)
                .map(normalize)
                .batch(batch_size=16)
            )
        )
        return self.model.evaluate(
            self.ds_test.map(process_path)
            .map(flatten_mask)
            .map(normalize)
            .batch(batch_size=16)
        )

    def predict(self, image):
        return self.model.predict(image)


if __name__ == "__main__":
    # Define parameters to be evaluated
    params = dict(vgg19=dict(hyper_params=dict()))

    # Run a trainer
    trainer = Trainer()
    result = trainer.train(params)
    logging.info(f"done! {result}")
