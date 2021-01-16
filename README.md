# Squid (the DP page view prediction ML model)

(last updated 12/2 8am ET)

### Why Squid
The messenger group chat the members created for this project was initially called *the DP Squad*. We got the inspiration of *Squid* from *Squad*.

### Motivation
The members of this project team have extensive connections with Penn’s student newspaper, The Daily Pennsylvanian (DP). Everyone at one point has worked as staff in the paper’s data analytics department; one member ran the department in 2019. During our time at the DP, we realized the potential of machine learning to improve the operations of the paper, which is increasingly moving towards a digital-first format as it struggles to find a viable business model in the twenty-first century. As it becomes less common for students to pick up a copy of the paper while moving around campus, the DP is striving to establish an online readership base in order to remain relevant to students and continue earning revenue from advertisers. This requires the paper to understand its audience and predict the performance of articles, which in turn requires machine learning models to provide this information. If the DP has a model that predicts the performance of news articles, they can use that information to form a social media posting strategy, curate their daily and weekly newsletters, and position articles on their website. This will help the paper increase their pageviews, keeping its work a part of the campus conversation and attracting more revenue opportunities.


### Problem Formulation

Classification:
1. Classify articles into the quintiles based on views (ranked wrt articles written in the same month)

Regression:
1. Predict article's percentile wrt articles written in the same month (reasoning: ultimately our purpose is to be able to rank articles published within a certain timeframe to decide what goes onto headline and what doesn't - David told us that model should be useful for ranking)

2. Predict article's views (reasoning: perhaps certain months just have bad articles so predicting actual views makes sense too)

### File Structure

Data: https://drive.google.com/drive/folders/14UH75BSa7sFZ17ZiX-kGvvOdsZJLeGYx?usp=sharing

**Deep Learning**

1. train-test-split.ipynb - merges formatted content and views together, runs the train and test split (stratified by year and month of publish), and creates train.csv and test.csv

2. Word Embeddings.ipynb - trains Word2Vec word embeddings on the content data, outputs 'word2vec_train2.txt' which is used for DL.

3. DL_Quintile_Classification.ipynb - initial Deep Learning file: loads existing word embeddings to create embedding matrix,trains RNN model to classify articles into quintiles, model evaluation (confusion matrix), explainable model insights using Eli5

4. DL_Percentile_Classification.ipynb - regression Deep Learning model to predict an article's views percentile (wrt articles published in that month)

_Deep Learning Results so far_

Model specification: GRU (32 units, 0.2 dropout), 0.25 dropout in embedding

1. Classification: ~39.1% accuracy the 5-quintile classification, 75.9% off-by-one accuracy
2. Regression on percentiles: mean absolute error of 16.6 percentile

_Deep Learning: To look into next_

1. Using titles to predict, instead of entire content
2. Regression on actual article views
3. Using LSTM (training overnight)
4. Using other Word Embedding methods - FastText (12/3 or 12/4)
5. Using Pre-trained Language Models - e.g. BERT, ELMO (12/3 or 12/4)
6. *Training DL models on the residuals from supervised learning steps (after accounting for seasonality, time spent in **headlines** and other factors)


**Supervised Learning**

Variables to used:
Topic probabilities (from LDA) AND/OR bag-of-words (count or TFIDF vectorizer), metadata (tags, duration on front page), temporal features (day published, month, year), author

Methods: Ridge regression / Logistic regression (simplest baseline), RF, GB, SVM (with hyperparameter tuning)

**Unsupervised Learning**

1. LDA_worksheet.ipynb - conducts LDA and NMF to get topics

_To do_
1. Retraining LDA on the training data (see Google Drive)
2. Selecting perfect number of topics (k) using topic coherence
3. Obtain topic probabilities for every article (to pass to Peter & James for supervised learning)


_Ideas in General_

1. How to incorporate time-varying effect of different topics on article views?
    - Idea: Run a rolling window sample (3 months) - and plot how the predicted parameters change over time (am especially interested to see how the topic probabilities or keyword importances change over time)

