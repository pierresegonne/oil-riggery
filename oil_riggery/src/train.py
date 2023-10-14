from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from oil_riggery import ROOT_PATH
from oil_riggery.src import config
from oil_riggery.src.dataset import NEPUDataset


def get_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    print("[INFO] loading dataset...")
    dataset = NEPUDataset.load(
        (ROOT_PATH / "data/NEPU_OWOD-1.0/JPEGImages").as_posix(),
        (ROOT_PATH / "data/NEPU_OWOD-1.0/Annotations").as_posix(),
    )
    train_dataset = dataset.get_train_dataset()
    eval_dataset = dataset.get_eval_dataset()
    test_dataset = dataset.get_test_dataset()

    def _generator_to_array(generator):
        X = []
        Y = []
        for x, y in generator:
            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)

    X_train, Y_train = _generator_to_array(train_dataset)
    X_eval, Y_eval = _generator_to_array(eval_dataset)
    X_test, Y_test = _generator_to_array(test_dataset)

    print("[INFO] train dataset shape: ", X_train.shape)

    return X_train, X_eval, X_test, Y_train, Y_eval, Y_test

def get_model() -> Model:
    # load the VGG16 network, ensuring the head FC layers are left off
    vgg = VGG16(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))
    # freeze all VGG layers so they will *not* be updated during the
    # training process
    vgg.trainable = False
    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bbox_head = Dense(128, activation="relu")(flatten)
    bbox_head = Dense(64, activation="relu")(bbox_head)
    bbox_head = Dense(32, activation="relu")(bbox_head)
    bbox_head = Dense(4, activation="sigmoid")(bbox_head)
    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bbox_head)

    print("[INFO] model input shape: ", model.input_shape)

    return model


def run(model: Model, X_train: np.ndarray, X_eval: np.ndarray, Y_train: np.ndarray, Y_eval: np.ndarray) -> None:
    opt = Adam(lr=config.INIT_LR)
    model.compile(loss="mse", optimizer=opt)
    print(model.summary())
    # train the network for bounding box regression
    print("[INFO] training bounding box regressor...")
    H = model.fit(
        X_train, Y_train,
        validation_data=(X_eval, Y_eval),
        batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHS,
        verbose=1)

    print("[INFO] saving object detector model...")
    model.save(config.MODEL_PATH, save_format="h5")
    # plot the model training history
    N = config.NUM_EPOCHS

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title("Bounding Box Regression Loss on Training Set")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(config.PLOT_PATH)


def main() -> None:
    X_train, X_eval, X_test, Y_train, Y_eval, Y_test = get_dataset()
    model = get_model()
    run(model, X_train, X_eval, Y_train, Y_eval)

if __name__ == "__main__":
    main()