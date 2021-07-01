from google.cloud import storage
import logging

from tensorflow.data.experimental import cardinality, AUTOTUNE
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
)
from tensorflow.keras.optimizers import Adam

from tensorboard.plugins.hparams import api as hp


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


def load_model_(model_name):
    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)
    # blob = bucket.blob("models/" + f"best_{model_name}.h5")
    # blob.download_to_filename(f"best_{model_name}.h5")

    return load_model(f"best_{model_name}.h5")


class Trainer(MLFlowBase):
    def __init__(self):
        super().__init__(EXPERIMENT_NAME, MLFLOW_URI)
        logging.basicConfig(level=logging.INFO)
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

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
            self.ds_train, self.ds_val, self.ds_test = holdout(df)

            # log params
            logging.info(f"Loading model: {model_name}")

            # create model
            model = get_model(model_name)

            # compling model and callbacks functions
            adam = Adam(learning_rate=hyper_params["learning_rate"], epsilon=0.1)

            model.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])

            # callbacks
            earlystopping = EarlyStopping(
                monitor="val_loss", mode="min", verbose=1, patience=30
            )

            # save the best model with lower validation loss
            checkpointer = ModelCheckpoint(
                filepath="seg_model.h5", verbose=1, save_best_only=True
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

            #
            hyper_p = hp.KerasCallback("logs/hparam_tuning", hyper_params)

            batch_size = 16
            ds_train = (
                self.ds_train.map(process_path, num_parallel_calls=AUTOTUNE)
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .shuffle(cardinality(self.ds_train))
                .batch(batch_size=batch_size)
                .prefetch(2)
            )

            ds_val = (
                self.ds_val.map(process_path, num_parallel_calls=AUTOTUNE)
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .batch(batch_size=batch_size)
            )

            history = model.fit(
                ds_train,
                epochs=100,
                callbacks=[
                    checkpointer,
                    earlystopping,
                    reduce_lr,
                    tensorboard,
                    hyper_p,
                ],
                validation_data=ds_val,
            )

            # save the trained model
            save_model(model, model_name)
            logging.info(f"best {model_name} saved")

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

    def predict(self, model_name="vgg19"):
        model: Model = load_model(model_name)
        return model.predict(
            self.ds_test.map(process_path).map(normalize), batch_size=16
        )


if __name__ == "__main__":
    # Define parameters to be evaluated
    params = dict(vgg19=dict(line_count=1_000, hyper_params=dict()))

    # Run a trainer
    trainer = Trainer()
    result = trainer.train(params)
    logging.info(f"done! {result}")
