import warnings, json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from tensorflow_addons.metrics import RSquare
from tensorflow.keras.metrics import RootMeanSquaredError

# Class for data feature compilation
class novozyme_model():

    def __init__(self, seqs, feature_limit=50):
        self.fl = feature_limit
        self.ids = []
        self.tms = []
        self.f = {}
        self.load_features(seqs)

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
            if len(idx) <= self.fl:
                self.f[idx] = []

    def append_data(self, id, seq, tm):
        self.ids.append(id)
        self.tms.append(tm)
        for idx in self.f:
            ct = seq.count(idx)
            self.f[idx].append(ct)
    
    def compile_dataset(self):
        self.Y = np.array(self.tms).astype('float32')
        self.X = np.array(
            [self.f[idx] for idx in self.f], dtype=object
        ).T.astype('float32')

    def dim_reduce(self, reduce_to=100):
        self.svd = TruncatedSVD(n_components=reduce_to, n_iter=10)
        self.X = self.svd.fit_transform(self.X)

    def build_model(self):

        # Input Layer
        model = tf.keras.Sequential([keras.Input(shape=(self.X.shape[1],))])

        # Try embedding
        model.add(keras.layers.Embedding(20, 1))

        # Hidden Layers
        model.add(keras.layers.Dense(units=2048, activation="relu"))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units=1024, activation="relu"))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(units=512, activation="relu"))

        # Output Layer and Model Training Params
        model.add(keras.layers.Dense(units=1, activation="linear"))
        optim = keras.optimizers.Adam(learning_rate=4.12e-4)
        loss_function = tf.losses.Huber(delta=1.0)
        model.compile(optimizer=optim,
              loss=loss_function,
              metrics=['mean_squared_error',
                        'mean_absolute_error',
                        RootMeanSquaredError(),
                        RSquare()])
        return model

    def train(self, epochs=10000, patience=3000, load_file=None, save_file=f'./novozyme.h5'):
        warnings.filterwarnings("ignore")
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        mc = tf.keras.callbacks.ModelCheckpoint(save_file, monitor='val_loss', save_best_only=True)
        # x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, shuffle=True)
        if load_file:
            self.model = load_model(load_file)
        else:
            self.model = self.build_model()
        self.model.fit(
            x=self.X,
            y=self.Y,
            # validation_data=(x_test, y_test),
            verbose=2,
            epochs=epochs,
            callbacks=[es, mc],
            # validation_split=0.2,
            steps_per_epoch=None,
            use_multiprocessing = True,
            batch_size=32
        )
        self.model = load_model(save_file)
        self.evaluate()

    def evaluate(self):
        self.eval = self.model.evaluate(self.X, self.Y, verbose=0)

    def predict(self, seqs):
        X = [[seq.count(idx) for idx in self.f] for seq in seqs]
        X = np.array(X).astype('float32')
        X = self.svd.transform(X)
        result = self.model.predict(X, verbose=0)
        return result