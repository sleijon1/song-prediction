import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.discriminant_analysis as skl_da
import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms
import sklearn.neighbors as skl_nb
import sklearn.preprocessing as skl_pre
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_rows", None, "display.max_columns", None)

data = pd.read_csv('data/training_data.csv')
print(data.head())
data.info()
data['label'].value_counts()
for i in data.columns:
    print(i, data[i].nunique())

# box plots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = \
plt.subplots(4, 3, figsize = (20, 20), constrained_layout=True)

fig.suptitle("Boxplots for several features", y = 1.03, fontsize = 18)

data.boxplot(column=["acousticness"], ax = ax1)
data.boxplot(column=["danceability"], ax = ax2)
data.boxplot(column=["duration"], ax = ax3)
data.boxplot(column=["energy"], ax = ax4)
data.boxplot(column=["instrumentalness"], ax = ax5)
data.boxplot(column=["key"], ax = ax6)
data.boxplot(column=["liveness"], ax = ax7)
data.boxplot(column=["loudness"], ax = ax8)
data.boxplot(column=["speechiness"], ax = ax9)
data.boxplot(column=["tempo"], ax = ax10)
data.boxplot(column=["valence"], ax = ax11)
data.boxplot(column=["time_signature"], ax = ax12)

#plt.show()

# histograms
# we have some features with long tails

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = \
plt.subplots(4, 3, figsize = (14, 10), constrained_layout=True)

data["acousticness"].hist(bins = 50, ax = ax1)
ax1.set_title("acousticness", y = 1.03, fontsize = 12)
ax1.set_yticklabels([])

data["danceability"].hist(bins = 50, ax = ax2)
ax2.set_title("danceability", y = 1.03, fontsize = 12)
ax2.set_yticklabels([])

data["duration"].hist(bins = 50, ax = ax3)
ax3.set_title("duration", y = 1.03, fontsize = 12)
ax3.set_yticklabels([])

data["energy"].hist(bins = 50, ax = ax4)
ax4.set_title("energy", y = 1.03, fontsize = 12)
ax4.set_yticklabels([])

data["instrumentalness"].hist(bins = 50, ax = ax5)
ax5.set_title("instrumentalness", y = 1.03, fontsize = 12)
ax5.set_yticklabels([])

data["liveness"].hist(bins = 50, ax = ax6)
ax6.set_title("liveness", y = 1.03, fontsize = 12)
ax6.set_yticklabels([])

data["loudness"].hist(bins = 50, ax = ax7)
ax7.set_title("loudness", y = 1.03, fontsize = 12)
ax7.set_yticklabels([])

data["speechiness"].hist(bins = 50, ax = ax8)
ax8.set_title("speechiness", y = 1.03, fontsize = 12)
ax8.set_yticklabels([])

data["tempo"].hist(bins = 50, ax = ax9)
ax9.set_title("tempo", y = 1.03, fontsize = 12)
ax9.set_yticklabels([])

data["valence"].hist(bins = 50, ax = ax10)
ax10.set_title("valence", y = 1.03, fontsize = 12)
ax10.set_yticklabels([])

data["key"].hist(bins = 50, ax = ax11)
ax11.set_title("key", y = 1.03, fontsize = 12)
ax11.set_yticklabels([])

data["time_signature"].hist(bins = 50, ax = ax12)
ax12.set_title("time_signature", y = 1.03, fontsize = 12)
ax12.set_yticklabels([])

#plt.show()

# searching for a good feature transformation
# log seems to be good for 'liveness'

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (18, 4), constrained_layout=False)

a = 0.001
data["liveness"].hist(bins = 20, ax = ax1)
ax1.set_title("liveness", fontsize = 14)
ax1.set_yticklabels([])

np.log(data["liveness"] + a).hist(bins = 20, ax = ax2)
ax2.set_title("log liveness", fontsize = 14)
ax2.set_yticklabels([])

(np.sign(data["liveness"]) * (data["liveness"].abs() ** (1/3))).hist(bins = 20, ax = ax3)
ax3.set_title("1/3 rt liveness", fontsize = 14)
ax3.set_yticklabels([])

(np.sign(data["liveness"]) * (np.sqrt(data["liveness"].abs()))).hist(bins = 20, ax = ax4)
ax4.set_title("sqrt liveness", fontsize = 14)
ax4.set_yticklabels([])

#plt.show()
data['log_liveness'] = np.log(data['liveness'] + a)

# searching for a good feature transformation
# 1/3 root seems to be good for 'loudness'
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (18, 4), constrained_layout=False)

a = 0.001
data["loudness"].hist(bins = 20, ax = ax1)
ax1.set_title("loudness", fontsize = 14)
ax1.set_yticklabels([])

# TODO data["loudness"] has large negative values
# log will give error when only adding a = 0.001
#np.log(data["loudness"] + a).hist(bins = 20, ax = ax2)
ax2.set_title("log loudness", fontsize = 14)
ax2.set_yticklabels([])

(np.sign(data["loudness"]) * (data["loudness"].abs() ** (1/3))).hist(bins = 20, ax = ax3)
ax3.set_title("1/3 rt loudness", fontsize = 14)
ax3.set_yticklabels([])

(np.sign(data["loudness"]) * (np.sqrt(data["loudness"].abs()))).hist(bins = 20, ax = ax4)
ax4.set_title("sqrt loudness", fontsize = 14)
ax4.set_yticklabels([])

#plt.show()

data['1/3rt_loudness'] = (np.sign(data["loudness"]) * (data["loudness"].abs() ** (1/3)))

# searching for a good feature transformation
# log is better than others for 'speechiness'
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (18, 4), constrained_layout=False)

a = 0.001
data["speechiness"].hist(bins = 20, ax = ax1)
ax1.set_title("speechiness", fontsize = 14)
ax1.set_yticklabels([])

np.log(data["speechiness"] + a).hist(bins = 20, ax = ax2)
ax2.set_title("log speechiness", fontsize = 14)
ax2.set_yticklabels([])

(np.sign(data["speechiness"]) * (data["speechiness"].abs() ** (1/3))).hist(bins = 20, ax = ax3)
ax3.set_title("1/3 rt speechiness", fontsize = 14)
ax3.set_yticklabels([])

(np.sign(data["speechiness"]) * (np.sqrt(data["speechiness"].abs()))).hist(bins = 20, ax = ax4)
ax4.set_title("sqrt speechiness", fontsize = 14)
ax4.set_yticklabels([])

#plt.show()

data['log_speechiness'] = np.log(data['speechiness'] + a)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize = (18, 4), constrained_layout=False)

a = 0.001
data["instrumentalness"].hist(bins = 20, ax = ax1)
ax1.set_title("speechiness", fontsize = 14)
ax1.set_yticklabels([])

np.log(data["instrumentalness"] + a).hist(bins = 20, ax = ax2)
ax2.set_title("log instrumentalness", fontsize = 14)
ax2.set_yticklabels([])

(np.sign(data["instrumentalness"]) * (data["instrumentalness"].abs() ** (1/3))).hist(bins = 20, ax = ax3)
ax3.set_title("1/3 rt instrumentalness", fontsize = 14)
ax3.set_yticklabels([])

(np.sign(data["instrumentalness"]) * (np.sqrt(data["instrumentalness"].abs()))).hist(bins = 20, ax = ax4)
ax4.set_title("sqrt instrumentalness", fontsize = 14)
ax4.set_yticklabels([])

#plt.show()

# scatter matrix
pd.plotting.scatter_matrix(data, alpha=0.2, figsize = (20,20))
#plt.show()

# Pearson correlation coefficient
data.corr()

data.head()


X = data[['acousticness', 'danceability', 'duration', 'energy', 'instrumentalness', 'key',\
          'liveness', 'loudness', 'mode', 'speechiness', 'tempo','time_signature', 'valence']]
y = data['label']


# creating dummies
# which features we assume as categorical?
# 'time_signature', 'key' or both?
X = pd.get_dummies(X, columns = ['time_signature', 'key'])
print(X.head())

# creating train-test split 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# normalizing the data is extremely important for some models
# for kNN as it is based on measuring the distance between the points
# if we have differently scalled data then some features will influence the model much more whereas other will
# not influence at all and it could become crutial

# minmaxscaler make all the features' values to be between 0 and 1
# it is done with the help of (x - min) / (max - min)

# it is vital to scale the data after train-test split to avoid any wrong results on the new data
# we also must scale them in the same way:
# we fit the scaler on training set and then use the same scaler to scale both train and test sets

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------- LOGREG ------------------- #
print("------------------- Logistic Regression -------------------")
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
THRESHOLD = 0.0
preds = np.where(log_reg.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)
score_like = accuracy_score(y_test, preds)
score_lr = log_reg.score(X_test, y_test)
cv_mean_lr = np.mean(cross_val_score(log_reg, X, y, cv=10))
print("Score (always predicting LIKE (log reg)):", score_like)
print("Score (log reg):", score_lr)
print("Cross validation mean score (log reg):", cv_mean_lr)
# ----------------- LOGREG ------------------- #

# ----------------- RANDOM FOREST ------------ #
print("------------------- Random forest -------------------")
ran_for = RandomForestClassifier()
ran_for.fit(X_train, y_train)
score_rf = ran_for.score(X_test, y_test)
cv_mean_rf = np.mean(cross_val_score(ran_for, X, y, cv=10))
print("Score (random forest):", score_rf)
print("Cross validation mean score (random forest):", cv_mean_rf)
# ----------------- RANDOM FOREST ------------ #
