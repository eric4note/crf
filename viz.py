
import numpy as np
import matplotlib.pyplot as plt
import pickle
from plot_cm import plot_confusion_matrix
from options import *

activities = [
    "A-Checking",
    "A-Moving",
    "A-Preparing",
    "A-Sitting",
    "A-Standing",
    "A-Taking-Dropping",
    "A-Transporting",
    "C-Compacting",
    "C-Leveling",
    "C-Placing",
    "C-Transporting",
    "F-Machining",
    "F-Placing-Fixing",
    "F-Transporting",
    "R-Machining",
    "R-Placing-Fixing",
    "R-Transporting"
]

cf = pickle.load(open(os.path.join(param_fdp_results, "results_{}_{}_cm.p".format(8, 0)), "rb"))

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cf, classes=range(1, len(activities) + 1), normalize=True,
                      title='Normalized confusion matrix, average accuracy: 98.77%')

# plt.show()
plt.gcf().savefig('confusion.png', dpi=300, bbox_inches='tight', pad_inches=0)



