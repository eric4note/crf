from pystruct.learners import OneSlackSSVM, NSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF
from options import *
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import time


X = []
Y = []

# a1, only use a1 features
# a2, only use a2 features
# all, use a1 and a2 features
pattern = "a2"

for vid in os.listdir(os.path.join(param_fdp_labels, pattern)):
    if os.path.splitext(vid)[1] == '.npd':
        x, y = pickle.load(open(os.path.join(param_fdp_labels, pattern, vid), "rb"))
        X.append(x)
        Y.append(y)

X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)

train_inds = []
test_inds = []

for i in range(len(X)):
    if i % 2 == 0:
        test_inds.append(i)
    else:
        train_inds.append(i)

train_inds = np.array(train_inds)
test_inds = np.array(test_inds)

X_train, X_test = X[train_inds], X[test_inds]
Y_train, Y_test = Y[train_inds], Y[test_inds]

start = time.time()
print(start)
# crf = EdgeFeatureGraphCRF(inference_method='qpbo')
# ssvm = OneSlackSSVM(crf, inference_cache=50, C=.1, tol=.1, switch_to='ad3', n_jobs=-1, verbose=4)

crf = EdgeFeatureGraphCRF()
ssvm = NSlackSSVM(crf, check_constraints=False, max_iter=50, batch_size=1, tol=0.1, n_jobs=-1, verbose=4)

ssvm.fit(X_train, Y_train)
pickle.dump(ssvm, open(os.path.join(param_fdp_labels, pattern, "ssvm.model"), "wb"))

Y_pred2 = ssvm.predict(X_test)
print("Results using also input features for edges")
print("Test accuracy: %.3f" % accuracy_score(np.hstack(Y_test), np.hstack(Y_pred2)))
print(confusion_matrix(np.hstack(Y_test), np.hstack(Y_pred2)))
print("Time used: {}".format(time.time()-start))











