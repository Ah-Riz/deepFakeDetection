from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import pickle
import os

path = ["eye_blink_data/DFD/manipulated/video_metrics.csv", "eye_blink_data/DFD/original/video_metrics.csv"]

data = None
for i in path:
    if data is None:
        data = np.genfromtxt(i, delimiter=',')[1:,1:]
    else:
        data = np.vstack((data, np.genfromtxt(i, delimiter=',')[1:,1:]))
print(data.shape)

target = data[:, -1]
data = data[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

with open('model/model_eye_blink.pkl', 'wb') as f:
    pickle.dump(clf, f)

os.makedirs('results/eye_blink', exist_ok=True)
np.savetxt('results/eye_blink/eye_blink_confusion_matrix.csv', confusion_matrix(y_test, y_pred), delimiter=',')