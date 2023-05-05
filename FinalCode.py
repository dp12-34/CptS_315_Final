import os    
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")
#used https://www.tensorflow.org/datasets/keras_example for my project, was a huge help
#for understanding what I needed to do
#https://stackoverflow.com/questions/64645503/tensorflow-datasets-cannot-batch-tensors-of-different-shapes-error-even-after-r
#


#grabs all my gpus (1) for helping to limit the memory growth
#using the GPU will make the training faster and prevent out of memory errors

#training pipeline
#creating a function that will normalize the pictures into the correct types
def normal_image(image, label):
   image = tf.image.resize(image, [180,180])
   return tf.cast(image, tf.float32) / 255., label


#initializing the testing and training datasets with parameters 
(ds_train, ds_test) = tfds.load(
    name ='stanford_dogs',
    split=['train', 'test'],
    as_supervised=True,

)

#mapping the training data for correct type
ds_train = ds_train.shuffle(1024)
ds_train = ds_train.map(normal_image, num_parallel_calls=tf.data.AUTOTUNE)

ds_train = ds_train.batch(128)


#building an evaluation pipeline
#testing pipeline
ds_test = ds_test.map(
    normal_image, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
#you cache after batching because you oculd have the same batch between epochs
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(180, 180)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test
)