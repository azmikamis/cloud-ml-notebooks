#!/usr/bin/env python
"""Script to illustrate usage of tf.estimator.Estimator in TF v1.8"""
import argparse
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.estimator import model_to_estimator


def serving_input_fn():
    inputs = {}
    inputs['x_input'] = tf.placeholder(shape=[None,28,28,1], dtype=tf.float32)
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def main(argv=None):
    # input image dimensions
    img_rows, img_cols = 28, 28
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    model = Sequential()
    model.add(Conv2D(20, (5, 5), activation='relu', padding='valid', input_shape=input_shape, name='x'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(40, (5, 5), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', name='probabilities'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(0.002),
                  metrics=['accuracy'])

    model_dir = "gs://mnist-estimator/train"
    est_mnist = model_to_estimator(keras_model=model, model_dir=model_dir)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x_input": x_train},
      y=y_train,
      batch_size=128,
      num_epochs=None,
      shuffle=True)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=500)
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x_input": x_test},
      y=y_test,
      num_epochs=1,
      shuffle=False)
    exporter = tf.estimator.FinalExporter('mnist', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=1, exporters=[exporter])
    #eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=1)
    
    tf.estimator.train_and_evaluate(est_mnist, train_spec, eval_spec)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
