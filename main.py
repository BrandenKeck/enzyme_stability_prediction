# Ability to specify GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Imports
import numpy as np
import pandas as pd

# Data import and slight manipulation
test_data = pd.read_csv('data/test.csv')
train_data = pd.read_csv('data/train.csv')
train_data['length']  = train_data['protein_sequence'].str.len()

# Experimenting with two model types
use_cnn1d_ann = False # Switch
if use_cnn1d_ann: # CNN1D-ANN Approach

    fll = 5
    ful = 50
    max_feature = 500
    train_data = train_data.query(f"length < {max_feature}")

    from models.cnn1D_ann_model import novozyme_model
    mod = novozyme_model(list(train_data.loc[:, 'protein_sequence']), 
                    list(train_data.loc[:, 'tm']), fll, ful, max_feature)

else: # CNN2D-RNN Approach

    size = 20 # NOTE: IMPORTANT
    train_data = train_data.query(f"length < {size*size}")

    from models.cnn_rnn_model import novozyme_model
    mod = novozyme_model(list(train_data.loc[:, 'protein_sequence']), 
                    list(train_data.loc[:, 'tm']), size)


# Train Independent Models on Combined Loss / Validation Loss
startpoints = 20
for i in np.arange(startpoints):
    mod.train(
        epochs=20, 
        patience=20, 
        vsplit=0.2,
        save_file = f'./saved/novozyme{i}.h5'
    )

# Evaluate the Trained Models Against Raw Loss for All Data
min_loss = 99999
best_mod = -1
from keras.models import load_model
for i in np.arange(startpoints):
    mod.model = load_model(f'./saved/novozyme{i}.h5')
    mod.evaluate()
    print(mod.eval[0])
    if mod.eval[0] < min_loss:
        print(f"Best Loss: {mod.eval[0]} for iteration {i}")
        min_loss = mod.eval[0]
        best_mod = i

# Load the best model
mod.model = load_model(f'./saved/novozyme{best_mod}.h5')

# Compile results
ids = [row['seq_id'] for index, row in test_data.iterrows()]
seqs = [row['protein_sequence'] for index, row in test_data.iterrows()]
tms = mod.predict(seqs)
tms = list(np.array(tms).reshape(-1))

# Save Submission File
results = pd.DataFrame({
    "seq_id": ids, 
    "tm": tms
})
results.to_csv("results.csv")