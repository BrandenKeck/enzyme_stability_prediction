# Imports
import warnings, math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from tensorflow_addons.metrics import RSquare
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.layers import (Embedding, Conv2D, Dropout, MaxPooling2D, 
                        Flatten, Reshape, Dense)

# Class for data feature compilation
class novozyme_model():

    def __init__(self, seq, tm, size=20):
        self.vocab_size = 20
        self.size = size
        self.X = [[one_hot(aa, self.vocab_size) for aa in s] for s in seq]
        self.X = pad_sequences(self.X, maxlen=self.size**2, padding='post')
        self.X = self.X.reshape((len(self.X), size, size)).astype('float32')
        self.Y = np.array(tm).astype('float32')

    def build_model(self):

        # Input Layer
        model = tf.keras.Sequential()

        # Params
        num_filters = 16
        conv_size = 2
        pool_size = 4
        dropout = 0.3
        dense_size = 256
        
        # CNN / Hidden Layers
        model.add(Embedding(20, num_filters))
        model.add(Reshape((self.size, self.size, num_filters)))
        model.add(Conv2D(num_filters, conv_size, input_shape=(self.size, self.size, num_filters), padding="same", activation=LeakyReLU()))
        model.add(Dropout(dropout))
        model.add(Conv2D(num_filters, conv_size, padding="same", activation=LeakyReLU()))
        model.add(Dropout(dropout))
        model.add(Conv2D(num_filters, conv_size, padding="same", activation=LeakyReLU()))
        model.add(MaxPooling2D(pool_size))
        model.add(Flatten())

        # Dense Hidden Layers
        model.add(Dense(units=dense_size, activation=LeakyReLU()))
        model.add(Dropout(dropout))
        model.add(Dense(units=dense_size/2, activation=LeakyReLU()))
        model.add(Dropout(dropout))
        model.add(Dense(units=dense_size/4, activation=LeakyReLU()))

        # Output Layer and Model Training Params
        model.add(Dense(units=1, activation=LeakyReLU()))
        optim = keras.optimizers.Adam(learning_rate=4.12e-4)
        loss_function = tf.losses.Huber(delta=1.0)
        model.compile(optimizer=optim,
              loss=loss_function,
              metrics=['mean_squared_error',
                        'mean_absolute_error',
                        RootMeanSquaredError(),
                        RSquare()])
        return model

    def train(self, epochs=10000, patience=3000, load_file=None, save_file=f'./novozyme.h5', validate=True):
        
        vsplit = 0.2
        warnings.filterwarnings("ignore")
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=vsplit, shuffle=True)

        if load_file: self.model = load_model(load_file)
        else: self.model = self.build_model()
        if validate:
            mons = 'val_loss'
            x = x_train
            y = y_train
            vdat = (x_test, y_test)
        else:
            mons = 'loss'
            x = self.X
            y = self.Y
            vdat = None
            vsplit = 0.0

        es = tf.keras.callbacks.EarlyStopping(monitor=mons, patience=patience)
        mc = tf.keras.callbacks.ModelCheckpoint(save_file, monitor=mons, save_best_only=True)
        self.model.fit(
            x=x,
            y=y,
            verbose=2,
            epochs=epochs,
            callbacks=[es, mc],
            validation_data=vdat,
            validation_split=vsplit,
            steps_per_epoch=None,
            use_multiprocessing = True,
            batch_size=32
        )
        self.model = load_model(save_file)
        self.evaluate()

    def evaluate(self):
        self.eval = self.model.evaluate(self.X, self.Y, verbose=0)

    def predict(self, seq):
        X = [[one_hot(aa, self.vocab_size) for aa in s] for s in seq]
        X = pad_sequences(X, maxlen=self.size**2, padding='post')
        X = X.reshape((len(X), self.size, self.size)).astype('float32')
        result = self.model.predict(X, verbose=0)
        return result