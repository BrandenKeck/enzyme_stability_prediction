# Imports
import warnings, json
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
from keras.layers import Input, Embedding, Reshape, Conv1D, Dropout, MaxPooling1D, Flatten, Dense, Concatenate

# Custom Loss Callback
class LossCallback(tf.keras.callbacks.Callback):

    def __init__(self, factor=0.5):
        self.factor = factor

    def on_epoch_end(self, epoch, logs={}):
        logs['loss_factor'] = self.factor * logs['val_loss'] + (1-self.factor) * logs['loss']

# Class for data feature compilation
class novozyme_model():

    def __init__(self, seq, tm, fll=10, ful=45, max_length=400):
        
        # Params
        self.vocab_size = 20
        self.fll = fll
        self.ful = ful
        self.max_length = max_length
        self.f = []

        # Data
        self.load_features(seq)
        self.X = np.array(seq)
        self.X1 = self.generate_x1(self.X)
        self.X2 = self.generate_x2(self.X)
        self.Y = np.array(tm).astype('float32')

    def load_features(self, seqs):

        # Generate substrings
        substring_counts={}
        try:
            substring_counts = json.load(open('sequences.json'))
        except:
            from difflib import SequenceMatcher
            for i in range(0, len(seqs)):
                for j in range(i+1,len(seqs)):
                    string1 = seqs[i]
                    string2 = seqs[j]
                    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
                    matching_substring=string1[match.a:match.a+match.size]
                    if(matching_substring not in substring_counts):
                        substring_counts[matching_substring]=1
                    else:
                        substring_counts[matching_substring]+=1
            with open('sequences.json', 'w') as fp:
                json.dump(substring_counts, fp)

        # Get interesting substrings as features
        for idx in substring_counts:
            if len(idx) >=self.fll and len(idx) <= self.ful:
                self.f.append(idx)

    def generate_x1(self, X):
        X1 = [[one_hot(aa, self.vocab_size) for aa in s] for s in X]
        X1 = pad_sequences(X1, maxlen=self.max_length, padding='post')
        X1 = X1.astype('float32')
        return X1

    def generate_x2(self, X):
        X2 = [[seq.count(idx) for idx in self.f] for seq in X]
        X2 = np.array(X2).astype('float32')
        return X2
    
    def build_model(self, num_filters = 16, conv_size = 3, pool_size = 4, dropout = 0.2, dense_size = 256):

        # Construct CNN1D Model
        cnn_input = Input(shape=(self.X1.shape[1], 1))
        cnn_emb = Embedding(20, 1)(cnn_input)
        cnn_reshape = Reshape((self.X1.shape[1], 1))(cnn_emb)
        cnn_conv1 = Conv1D(filters=num_filters, kernel_size=conv_size, activation=LeakyReLU())(cnn_reshape)
        cnn_drop1 = Dropout(dropout)(cnn_conv1)
        cnn_conv2 = Conv1D(filters=num_filters, kernel_size=conv_size, activation=LeakyReLU())(cnn_drop1)
        cnn_drop2 = Dropout(dropout)(cnn_conv2)
        cnn_conv3 = Conv1D(filters=num_filters, kernel_size=conv_size, activation=LeakyReLU())(cnn_drop2)
        cnn_pool = MaxPooling1D(pool_size)(cnn_conv3)
        cnn_flat = Flatten()(cnn_pool)
        cnn_dense1 = Dense(units=dense_size, activation=LeakyReLU())(cnn_flat)
        cnn_outlayer = Dense(units=dense_size/2, activation=LeakyReLU())(cnn_dense1)

        # Construct ANN Model
        ann_input = Input(shape=(self.X2.shape[1],))
        ann_emb = Embedding(20, 1)(ann_input)
        ann_flat = Flatten()(ann_emb)
        ann_dense1 = Dense(units=dense_size, activation=LeakyReLU())(ann_flat)
        ann_drop1 = Dropout(dropout)(ann_dense1)
        ann_dense2 = Dense(units=dense_size/2, activation=LeakyReLU())(ann_drop1)
        ann_drop2 = Dropout(dropout)(ann_dense2)
        ann_outlayer = Dense(units=dense_size/2, activation=LeakyReLU())(ann_drop2)

        # Construst Concatinated Model
        cnn_ann_concat = Concatenate()([cnn_outlayer, ann_outlayer])
        concat_dense1 = Dense(units=dense_size/2, activation=LeakyReLU())(cnn_ann_concat)
        concat_drop1 = Dropout(dropout)(concat_dense1)
        concat_dense2 = Dense(units=dense_size/4, activation=LeakyReLU())(concat_drop1)
        concat_drop2 = Dropout(dropout)(concat_dense2)
        concat_dense3 = Dense(units=dense_size/8, activation=LeakyReLU())(concat_drop2)
        concat_dense4 = Dense(units=dense_size/16, activation=LeakyReLU())(concat_dense3)
        outlayer = Dense(units=1, activation="linear")(concat_dense4)

        # Generate the final model
        model = tf.keras.Model(inputs = [cnn_input, ann_input], outputs = outlayer)

        # Output Layer and Model Training Params
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=4.12e-3,
            decay_steps=250,
            decay_rate=0.96
        )
        optim = keras.optimizers.Adam(learning_rate=lr_schedule)
        loss_function = tf.losses.MeanSquaredError()
        model.compile(optimizer=optim,
              loss=loss_function,
              metrics=['mean_squared_error',
                        'mean_absolute_error',
                        RootMeanSquaredError(),
                        RSquare()])

        return model

    def train(self, epochs=10000, patience=3000, load_file=None, save_file=f'./novozyme.h5', validate=True, vsplit=0.2):
        
        warnings.filterwarnings("ignore")
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=vsplit, shuffle=True)

        if load_file: self.model = load_model(load_file)
        else: self.model = self.build_model()
        if validate:
            mons = 'loss_factor'
            x1 = self.generate_x1(x_train)
            x2 = self.generate_x2(x_train)
            y = y_train
            vdat = ([self.generate_x1(x_test), self.generate_x2(x_test)], y_test)
        else:
            mons = 'loss'
            x1 = self.generate_x1(self.X)
            x2 = self.generate_x2(self.X)
            y = self.Y
            vdat = None
            vsplit = 0.0

        lcb = LossCallback()
        es = tf.keras.callbacks.EarlyStopping(monitor=mons, patience=patience)
        mc = tf.keras.callbacks.ModelCheckpoint(save_file, monitor=mons, save_best_only=True)
        self.model.fit(
            x=[x1, x2],
            y=y,
            verbose=1,
            epochs=epochs,
            callbacks=[lcb, es, mc],
            validation_data=vdat,
            validation_split=vsplit,
            steps_per_epoch=None,
            use_multiprocessing = True,
            batch_size=32
        )
        self.model = load_model(save_file)

    def evaluate(self):
        x1 = self.generate_x1(self.X)
        x2 = self.generate_x2(self.X)
        y = self.Y
        self.eval = self.model.evaluate(
            x=[x1, x2],
            y=y,
            verbose=0
        )

    def predict(self, seq):
        X1 = self.generate_x1(seq)
        X2 = self.generate_x2(seq)
        result = self.model.predict([X1, X2], verbose=0)
        return result