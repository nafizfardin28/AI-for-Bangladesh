import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = 224



def load_data():
    data = []
    labels = []


    for img_name in os.listdir("cropped_faces/real"):
        img_path = os.path.join("cropped_faces/real", img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(0)

    for img_name in os.listdir("cropped_faces/fake"):
        img_path = os.path.join("cropped_faces/fake", img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(1)

    data = np.array(data) / 255.0
    labels = np.array(labels)

    print("Total images loaded:", len(data))

    return train_test_split(data, labels, test_size=0.2, random_state=42)


def build_model():
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    model.build((None, 224, 224, 3))

    return model


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data()


    model = build_model()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )


    class_weight = {0: 1.0, 1: 1.3}  

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=8,
        callbacks=[early_stop],
        class_weight=class_weight
)


    loss, accuracy = model.evaluate(X_test, y_test)
    print("\nTest Accuracy:", accuracy)


    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.45).astype("int32")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    model.save("models/deepfake_model.keras")

    print("\nModel saved successfully!")