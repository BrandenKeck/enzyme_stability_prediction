# Enzyme Stability Prediction  
### Novozymes Enzyme Stability Prediction Contest Submission  
The goal of this analysis is to accurately predict enzyme stability from the Novozymes dataset  
https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/overview

# Methods  
### Scoring  
Kaggle scoring for this competition was based on Spearman's rank correlation coefficient.  To prevent leaderboard abuse, all teams are shown a "public" score with each submission (this is the Spearman's rank correlation coefficient for half of the test data).  The "private" score for each team is revealed at the competition deadline (this is the Spearman's rank correlation coefficient for the entire test data).  Two submissions with respective "public" scores of 0.103 and 0.054 were submitted.  Their resulting "private" / complete scores were 0.065 and 0.087 respectively.  

### Approach  
Protein sequence data can be used in a variety of ways.  Features can include base sequence components, computed 3D structure components, computed chemical properties, or other components derived from molecular modeling of the structure.  This competition was entered with 1 week remaining until deadline.  As such, only the sequence features themselves were considered.  

#### ANN Approach  
A simple ANN was constructed using presence of substrings as a feature.  Maximum substring length was considered a hyper parameter.  This approach seemed to yield positive correlation in the "public" test data, but was ultimately shown to yeild no predictive capabilities when the "private" results were released.  

#### RNN Approach  
Two RNN submissions were created.  To create the RNN, protein sequences were padded with max sequence length as a hyper parameter.  Each submission leaveraged LSTM units which proved difficult to train.  Predicted results were identical for all sequences in the test data (supposedly due to underfitting).  As such these submissions were invalid and no score was achieved.  

#### 2D CNN Approach  
Protein sequences were padded (with max sequence size as a hyper parameter) and reshaped into square 2D feature maps.  A typical CNN was then trained against the transformed data.  These results were highly variable.  Two attempts yielded positive public and private scores.  However, three attempts yielded near-zero correlation.  

#### 1D CNN Approach  
Given the difficultly training LSTM RNN units, a 1D CNN was tested to attempt to extract features from padded sequences (with max sequence size as a hyper parameter).  Many submissions were made with this approach.  Results of this approach were more variable, but largely more accurate than the 2D CNN approach.  "Private" score results showed 4 postive correlation predictions and 2 attempts with near-zero correlation.  

#### 2D CNN / RNN Series Hybrid Approach  
The first attempted hybrid approach was that of a 2D CNN (with data preprocessed as described above) combined in series with an RNN.  Given the difficulty training LSTM units previously, GRUs were used instead.  This approach was abandoned after 1 attempt, but yielded the highest "private" score of all approaches used.  If selected, leadboard position would have improved by ~50.  

#### 1D CNN / ANN Parallel Hybrid Approach  
This approach was the main final focuse of this project.  The previously described ANN and 1D CNN techniques were combined in parallel to attempt to take advanatage of both substructure presence and sequence order when making predictions.  The results were promising.  All "private" scores from this approach were positive correlations, included the second highest score overall.  

#### Late Stage Fine-Tuning Approaches  
As each method was developed, additional fine-tuning approachs were implemented, which somewhat contributed to higher performance in the latter approaches.  In addition to parameter tweaking, three techniques were used to experimentally overcome the supposed disparity between the availble training data and the target submission dataset.  For example, the training data contained sequences that drastically varied in size and composition whereas the submission data consisted of sequences all of length ~200 that essentially varied due to point mutations.  
  
The first "advanced" method used to circumvent this problem was learning rate scheduling, which was used to increase the chances of finding a global optimum by using a large learning rate to start and slowly decreasing the learning rate as training progressed.  Additionally, a custom metric was implemented to facilitate "early stopping" and "model checkpoint".  This metric combined the "loss" and "val_loss" components in the model into a balanced factor: $(\alpha) \times val_loss + (1-\alpha) \times loss$, where $\alpha$ is a user-defined hyperparameter.  This was implemented because training (stopping/checkpointing) without validation was resulting in overfitting, but training (stopping/checkpointing) on validation alone would result in very poor fits.  Finally, given the variability in the results from these first two methods, a multi-simulation approach was implemented to first train (checkpoint) several models based on the custom loss factor, then select from the results the best overall loss on the entire training set.  
  
Upon implementing these methods, most submissions resulted in positive Spearman correlations.  These methods were mostly used to improve the 1D CNN / ANN parallel model.  In hindsight, using these methods on the 2D CNN / RNN (GRU) model may have resulted in a much better overall fit.  
  
# Results  
### Final Score  
Final Rank: 1664 / 2543  
Final Score: 0.087 / 0.545 (Best Submitted Spearman Correlation - "Private" Score)  

### Best Submitted Score (in Hindsight)  
Best Score: 0.108 / 0.545
