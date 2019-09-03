from pystruct.learners import OneSlackSSVM, NSlackSSVM
from pystruct.models import EdgeFeatureGraphCRF
from options import *
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import csv

results = []
# k nearest neighbors
for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]:
    print("K: {} nearest neighbors".format(k))
    # activity features
    for pattern in range(3):
        print("Level {} activity features".format(pattern))

        X = []
        Y = []

        for vid_ in sorted(os.listdir(param_fdp_videos)):
            vid = vid_[:-4]
            x, y = pickle.load(open(os.path.join(param_fdp_results, "{}_{}_{}.npd".format(k, pattern, vid)), "rb"))
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

        crf = EdgeFeatureGraphCRF()
        ssvm = NSlackSSVM(crf, check_constraints=False, max_iter=50, batch_size=1, tol=0.1, n_jobs=-1, verbose=3)

        print("Training started")
        start = time.time()

        ssvm.fit(X_train, Y_train)
        train_time = time.time() - start

        print("Training completed and testing started")
        print("Time used for training: {}".format(train_time))

        pickle.dump(ssvm, open(os.path.join(param_fdp_results, "results_{}_{}_model.p".format(k, pattern)), "wb"))

        start = time.time()
        Y_pred = ssvm.predict(X_test)
        test_time = time.time()-start

        acc = accuracy_score(np.hstack(Y_test), np.hstack(Y_pred))
        cm = confusion_matrix(np.hstack(Y_test), np.hstack(Y_pred))
        pickle.dump(cm, open(os.path.join(param_fdp_results, "results_{}_{}_cm.p".format(k, pattern)), "wb"))

        print("Test completed with an accuracy: %.3f" % acc)
        print(cm)
        print("Time used for testing: {}".format(test_time))

        results.append([k, pattern, acc, train_time, test_time, len(X_train), len(X_test)])

with open(os.path.join(param_fdp_results, "log.csv".format(k, pattern)), "w") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(["K", "act_feature", "acc", "time_train", "time_test", "X_train", "X_test"])
    for r in results:
        wr.writerow(r)












