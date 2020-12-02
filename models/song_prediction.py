import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.discriminant_analysis as skl_da
import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms
import sklearn.neighbors as skl_nb
import sklearn.preprocessing as skl_pre
import sklearn.discriminant_analysis as skl_da
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
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


#X = data[['acousticness', 'danceability', 'duration', 'energy', 'instrumentalness', 'key',\
#          'liveness', 'loudness', 'mode', 'speechiness', 'tempo','time_signature', 'valence']]
X = data[['acousticness', 'danceability', 'energy', \
          'loudness', 'speechiness', 'valence']]
y = data['label']


# creating dummies (if line 211+212 is used)
#X = pd.get_dummies(X, columns = ['time_signature', 'key'])
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

scaler = MinMaxScaler()
scaler.fit(X_train)

X = scaler.transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# initial models
log_reg = LogisticRegression(solver='liblinear', penalty='l1')
ran_for = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors = 5)


# cross validation to compare models on the same data, using the same features
n_fold = 10

models = []
models.append(log_reg)
models.append(ran_for)
models.append(knn)
models.append(skl_da.LinearDiscriminantAnalysis())
models.append(skl_da.QuadraticDiscriminantAnalysis())


for fig_num in plt.get_fignums():
	plt.close(fig_num)


conf_fig, ((ax1, ax2, ax3,), (ax4, ax5, ax6)) = \
plt.subplots(2, 3, figsize=(20, 20), constrained_layout=True)

axis = [ax1, ax2, ax3, ax4, ax5]

for m in range(np.shape(models)[0]):
	model = models[m]
	model.fit(X_train, y_train)
	prediction = model.predict(X_test)
	#conf_matrix = confusion_matrix(y_test, prediction)
	plot_conf_matrix = plot_confusion_matrix(model, X_test, y_test, ax=axis[m]) 
	plot_conf_matrix.ax_.set_title(model.__class__.__name__)
ax6.axis("off")

plt.show()


misclassification = np.zeros((n_fold, len(models)))
cv = skl_ms.KFold(n_splits=n_fold, random_state=1, shuffle=True)

for i, (train_index, val_index) in enumerate(cv.split(X)):
	X_train, X_val = X[train_index], X[val_index]
	y_train, y_val = y[train_index], y[val_index]

	for m in range(np.shape(models)[0]):
		model = models[m]
		model.fit(X_train, y_train)
		prediction = model.predict(X_val)
		misclassification[i,m] = np.mean(prediction != y_val)

myfig, ax1 = plt.subplots() 
ax1.boxplot(misclassification)
ax1.set_title('Cross Validation Error')
ax1.set_ylabel('Validation Error')
ax1.set_xlabel('Model')


plt.xticks(np.arange(len(models))+1, ('logReg','ranFor','k-NN', 'LDA', 'QDA'))
plt.show()



# Actual classifying
data_classify = pd.read_csv('data/songs_to_classify.csv')

X_classify = data_classify[['acousticness', 'danceability', 'energy', \
          'loudness', 'speechiness', 'valence']]

ran_for = RandomForestClassifier()

scaler = MinMaxScaler()
scaler.fit(X)

X = scaler.transform(X)
X_classify = scaler.transform(X_classify)
ran_for.fit(X,y)
prediction = ran_for.predict(X_classify)

mystr = ""
for val in prediction:
	mystr += str(val)
print("Classifying 200 songs: \n" + mystr)


"""
# ----------------- LOGREG ------------------- #
print("------------------- Logistic Regression -------------------")
log_reg = LogisticRegression(solver='liblinear', penalty='l1')
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
#print("------------------- Random forest -------------------")
#ran_for = RandomForestClassifier()
#ran_for.fit(X_train, y_train)
#score_rf = ran_for.score(X_test, y_test)
#cv_mean_rf = np.mean(cross_val_score(ran_for, X, y, cv=10))
#print("Score (random forest):", score_rf)
#print("Cross validation mean score (random forest):", cv_mean_rf)
# ----------------- RANDOM FOREST ------------ #
"""