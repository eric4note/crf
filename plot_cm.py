import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fs_txt = 9
    fs_title_label = 9

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fs_title_label)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fs_txt)

    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes, fontsize=fs_txt)
    plt.yticks(tick_marks, classes, fontsize=fs_txt)

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=fs_txt)

    plt.ylabel('True activity no.', fontsize=fs_title_label)
    plt.xlabel('Predicted activity no.', fontsize=fs_title_label)
    plt.tight_layout()

    # fig = plt.gcf()
    # fig.set_size_inches(12, 12)
    # plt.tight_layout()
    # fig.savefig(os.path.join(src, "split_{}.tif".format(split)), dpi=300)