import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.utils.class_weight import compute_class_weight

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_electro_model():
    # Config
    BATCH_SIZE = 32
    EPOCHS = 30
    IMAGE_SIZE = (224, 224)
    DATASET_DIR = "/Users/mac/Downloads/Electronic Accessories Classification Dataset A Comprehensive Collection for Accessory Recognition and Categorization/Electronic Accessories Classification Dataset A Comprehensive Collection for Accessory Recognition and Categorization/Original"

    # Load dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='training'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=123,
        validation_split=0.2,
        subset='validation'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Class weights
    labels = np.concatenate([y for x, y in train_ds], axis=0)
    class_indices = np.argmax(labels, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
    class_weight_dict = dict(enumerate(class_weights))

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.2),
    ])

    # Base model without drop_connect_rate
    base_model = EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # Model architecture
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "best_model.keras"),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        class_weight=class_weight_dict
    )

    # Save
    model.save(os.path.join(MODELS_DIR, "electro_accessory_model.keras"))
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(os.path.join(MODELS_DIR, "electro_accessory_model.tflite"), 'wb') as f:
        f.write(tflite_model)
    print("Model conversion complete")

if __name__ == "__main__":
    train_electro_model()