from google.cloud import storage
import logging
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)
from tensorflow.keras.optimizers import Adam

from MRIsegmentation.data import get_data, get_data_from_drive, holdout
from MRIsegmentation.params import BUCKET_NAME, EXPERIMENT_NAME, MLFLOW_URI
from MRIsegmentation.pipeline import get_pipeline
from MRIsegmentation.model import get_model
from MRIsegmentation.mlflow import MLFlowBase
from MRIsegmentation.utils import (
    focal_tversky,
    tversky,
    tversky_loss,
    process_path,
    normalize,
)


def save_model(best_model: Model, model_name: str):
    best_model.save(f"best_{model_name}.h5")

    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)
    # blob = bucket.blob("models/" + f"best_{model_name}.h5")
    # blob.upload_from_filename(f"best_{model_name}.h5")


def load_model(model_name):
    print("--- ", model_name)
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob("models/" + f"best_{model_name}.h5")
    blob.download_to_filename(f"best_{model_name}.h5")

    return load_model(
        f"best_{model_name}.h5",
        custom_objects={
            "focal_tversky": focal_tversky,
            "tversky": tversky,
            "tversky_loss": tversky_loss,
        },
    )


class Trainer(MLFlowBase):
    def __init__(self):
        super().__init__(EXPERIMENT_NAME, MLFLOW_URI)
        logging.basicConfig(level=logging.INFO)

    def train(self, params):
        # iterate on models
        for model_name, model_params in params.items():

            line_count = model_params["line_count"]
            hyper_params = model_params["hyper_params"]

            # create a mlflow training
            self.mlflow_create_run()

            # log params
            self.mlflow_log_param("model_name", model_name)
            self.mlflow_log_param("line_count", line_count)
            for key, value in hyper_params.items():
                self.mlflow_log_param(key, value)

            # get data
            df = get_data_from_drive()
            logging.info(f"Data loaded: {df.shape}")

            # holdout
            ds_train, ds_val, ds_test = holdout(df)

            # log params
            logging.info(f"Loading model: {model_name}")

            # create model
            model = get_model(model_name)

            # compling model and callbacks functions
            adam = Adam(lr=0.05, epsilon=0.1)

            model.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])

            # callbacks
            earlystopping = EarlyStopping(
                monitor="tversky_loss", mode="min", verbose=1, patience=30
            )

            # save the best model with lower validation loss
            checkpointer = ModelCheckpoint(
                filepath="seg_model.h5", verbose=1, save_best_only=True
            )

            # reduce learning rate when on a plateau
            reduce_lr = ReduceLROnPlateau(
                monitor="tversky_loss",
                mode="min",
                verbose=1,
                patience=10,
                min_delta=0.0001,
                factor=0.2,
            )

            batch_size = 16
            ds_train = (
                ds_train.map(process_path).map(normalize).batch(batch_size=batch_size)
            )

            history = model.fit(
                ds_train,
                epochs=60,
                callbacks=[checkpointer, earlystopping, reduce_lr],
                validation_data=ds_val,
            )

            # save the trained model
            save_model(model, model_name)
            logging.info(f"best {model_name} saved")

            # push best params & score to mlflow
            # for k, v in grid_search.best_params_.items():
            #    self.mlflow_log_param('best__' + k, v)

            # push metrics to mlflow
            self.mlflow_log_metric("loss", history.history["loss"])
            self.mlflow_log_metric("val_loss", history.history["val_loss"])
            self.mlflow_log_metric("tversky", history.history["tversky"])
            self.mlflow_log_metric("val_tversky", history.history["val_tversky"])

            # return the gridsearch in order to identify the best estimators and params

        return f"goto {MLFLOW_URI}/#/experiments/{self.mlflow_experiment_id}"

    def predict(self, df, model_name="vgg19"):
        print(model_name)
        model = load_model(model_name)
        return model.predict(df)


if __name__ == "__main__":
    # Define parameters to be evaluated
    params = dict(vgg19=dict(line_count=1_000, hyper_params=dict()))

    # Run a trainer
    trainer = Trainer()
    result = trainer.train(params)
    logging.info(f"done! {result}")
