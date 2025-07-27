import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import keras
import tensorflow as tf
import joblib

# ----------------- Load Image Paths --------------------
def data(base_dataset_path, split_folder):
    images, labels = [], []
    for fracture_type_folder in os.listdir(base_dataset_path):
        fracture_type_path = os.path.join(base_dataset_path, fracture_type_folder)
        if not os.path.isdir(fracture_type_path):
            continue
        split_path = os.path.join(fracture_type_path, split_folder)
        if os.path.isdir(split_path):
            for image_filename in os.listdir(split_path):
                if image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(split_path, image_filename)
                    images.append(image_path)
                    labels.append(fracture_type_folder)
    df = pd.DataFrame({'image': images, 'label': labels})
    return df

# ----------------- Paths --------------------
BASE_DATA_PATH = 'Bone Break Classification'

# Load data
train_df = data(BASE_DATA_PATH, split_folder='train')
print(f"Loaded {len(train_df)} training images.")

full_test_df = data(BASE_DATA_PATH, split_folder='test')
print(f"Loaded {len(full_test_df)} test images.")

# Split test into val/test
val_df, test_df = train_test_split(full_test_df, test_size=0.5, stratify=full_test_df['label'], random_state=42)
print(f"{len(val_df)} for validation, {len(test_df)} for final testing.")

# ----------------- Data Generator --------------------
image_size = (224, 224)
batch_size = 32
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    train_df, x_col='image', y_col='label',
    target_size=image_size, batch_size=batch_size,
    class_mode='categorical', shuffle=True
)

idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
joblib.dump(idx_to_class, 'idx_to_class.pkl')

val_generator = datagen.flow_from_dataframe(
    val_df, x_col='image', y_col='label',
    target_size=image_size, batch_size=batch_size,
    class_mode='categorical', shuffle=False
)

test_generator = datagen.flow_from_dataframe(
    test_df, x_col='image', y_col='label',
    target_size=image_size, batch_size=batch_size,
    class_mode='categorical', shuffle=False
)

# ----------------- Class Weights --------------------
classes = train_generator.class_indices
class_names = list(classes.keys())
class_weights = compute_class_weight('balanced', classes=class_names, y=train_df['label'])
class_weights = dict(enumerate(class_weights))

# ----------------- Model --------------------
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2,2)),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Dropout(.3),

    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Dropout(.3),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(.3),
    keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint = tf.keras.callbacks.ModelCheckpoint("model.h5", save_best_only=True, monitor="val_loss")

print("\nStarting model training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)
print("Training finished.")

# ----------------- Save Model --------------------
model.save("model.h5")

# ----------------- Evaluate Model --------------------
print("\nEvaluating on test data...")
loss, acc = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# ----------------- Prediction Distribution --------------------
preds = model.predict(test_generator)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(test_generator.labels, axis=0) if test_generator.class_mode == 'categorical' else test_generator.labels

print("\nPrediction Class Distribution:")
print(pd.Series(pred_labels).value_counts())

print("\nClassification Report:")
print(classification_report(test_generator.classes, pred_labels, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(test_generator.classes, pred_labels))
