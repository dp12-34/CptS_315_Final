import io
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    img = tf.image.resize(img,[180,180])
    return (img, label)
    

train_dataset, test_dataset = tfds.load(name="stanford_dogs", split=['train', 'test'], as_supervised=True)

train_dataset = train_dataset.shuffle(1024)
train_dataset = train_dataset.map(_normalize_img)

test_dataset = test_dataset.batch(32)
train_dataset = train_dataset.map(_normalize_img)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,2,padding='same',activation='relu',input_shape=(180,180,3)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32,2,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120,activation='softmax')
])
train_dataset = tf.expand_dims(train_dataset, axis=-1)


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy')


history = model.fit(
    train_dataset,
    epochs=5)