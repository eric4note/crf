"""
create dataset
"""

from options import *
import pickle
import pandas as pd
import numpy as np
import math
import csv

def prepare_data(pattern, summary_clips=False):
    """
    1. create a dataset in the following format:
    {
        vid: [[sa1, sa2, sb, sy], ...]
            where each sa1 consists of n_workers action feature vectors, each of which is in the shape of (17,),
            each sa2 consists of n_workers activity feature vectors, each of which is in the shape of (12,),
            each sb consists of n_workers bounding boxes, each of which is in the shape of (4,), and
            each sy consists of n_workers a2 indexes, each of which is an integer.
        ...
    }

    2. save it with
    pickle.dump(dataset, open(os.path.join(param_fdp_labels, "all_x1_x2_xb_y.pickle"), "wb"))
    """

    # *.a1 and *.a2 have all actions and activity results
    # recognized using C3D based on tracklet information
    # without creating images on local hard drivers, but in memory!
    # for vid_ in sorted(os.listdir(param_fdp_videos)):


    fp_test_list = os.path.join(param_fdp_data, "splits", "testlist01.txt")
    f = open(fp_test_list)
    try:
        lines = f.readlines()
        # for line in lines:
        #     print(line)
    finally:
        f.close()

    gts_all = {}
    for line in lines:
        # activity/vid-segid-workerid
        fds = line.split('/')
        act = fds[0]
        act_id = activities.index(act)
        cnames = fds[1].split('-')
        vid = cnames[0]
        seg_no = int(cnames[1])
        worker_id = int(cnames[2][:-1])
        if vid in gts_all:
            gts_all[vid].append([act_id, seg_no, worker_id])
        else:
            gts_all[vid] = [[act_id, seg_no, worker_id]]

    for k, v in gts_all.items():
        gts_all[k] = pd.DataFrame(gts_all[k], columns=['a2_ind', 'seg_no', 'wkr_id'])


    dataset = {}
    for vid_ in sorted(os.listdir(param_fdp_videos)):
        vid = vid_[:-4]
        gts = gts_all[vid]

        # load segments from *.a1 files. Although this information can be obtained from
        # *.seg files, action recognition results cannot be assessed from them.
        vid = vid_[:-4]
        fp_a1 = os.path.join(param_fdp_segments, vid + ".a1")
        with open(fp_a1, 'rb') as f_a1:
            a1s = pickle.load(f_a1)

        # *.a2 has the same data structure with *.a1, only with different labels.
        # a1 falls into [0, 16], while a2 falls into [0, 11]
        fp_a2 = os.path.join(param_fdp_segments, vid + ".a2")
        with open(fp_a2, 'rb') as f_a2:
            a2s = pickle.load(f_a2)

        assert (len(a1s) == len(a2s))

        subset = []
        last_seg_no = a1s[0]['segment'][0]
        n_segs = len(a1s)
        sx1, sx2, sxb, sy = [], [], [], []

        for k, [a1, a2] in enumerate(zip(a1s, a2s)):
            # find the tracklets via filtering in dataframe
            gt = gts[(gts['seg_no'] == a1['segment'][0]) & (gts['wkr_id'] == a1['segment'][1])]

            # if found one
            if len(gt) == 1:
                # create a new sample if the seg_no changes
                if a1['segment'][0] != last_seg_no:
                    # there should be at least two workers
                    if len(sx1) > 1:
                        subset.append([last_seg_no, sx1, sx2, sxb, sy])
                        print("Added a sample with {} workers at segment {}".format(len(sx1), last_seg_no))

                    last_seg_no = a1['segment'][0]
                    sx1, sx2, sxb, sy = [], [], [], []

                sx1.append(a1['scores'])  # a1_feature
                sx2.append(a2['scores'])  # a2_feature
                sxb.append(a1['segment'][2:6])  # box
                sy.append(gt.iloc[0, 0])  # a2_ind

                # come to the last seg
                if k == n_segs - 1:
                    # there should be at least two workers
                    if len(sx1) > 1:
                        subset.append([last_seg_no, sx1, sx2, sxb, sy])
                        print("Added a sample with {} workers at segment {}".format(len(sx1), last_seg_no))
                    print("=========================")
            # else:
            #     print("tracklet segno_workerid: {}_{} not found".format(a1['segment'][0], a1['segment'][1]))

        if summary_clips:
            for _, _, _, sxb, sy in subset:
                for box, y in zip(sxb, sy):
                    if y in dataset:
                        dataset[y].append([box[2]-box[0], box[3]-box[1]])
                    else:
                        dataset[y] = [[box[2]-box[0], box[3]-box[1]]]

        X = []
        Y = []
        for seg_no, sx1, sx2, sxb, sy in subset:
            sub_edges = []
            sub_edge_features = []

            for i in range(len(sxb)):
                max_sp = 0
                another_end = -1
                for j in range(len(sxb)):
                    if i != j:
                        sp = sprel(sxb[i], sxb[j])
                        if sp > max_sp:
                            max_sp = sp
                            another_end = j

                sub_edges.append(np.array([i, another_end], dtype=int))
                ef = np.zeros(NA2)
                ef[sy[another_end]] = max_sp
                sub_edge_features.append(ef)

            if pattern == "a1":
                xs = [np.array(x) for x in sx1]
            elif pattern == "a2":
                xs = [np.array(x) for x in sx2]
            else:
                xs = [np.hstack((np.array(x1), np.array(x2))) for x1, x2 in zip(sx1, sx2)]

            if len(xs) > 0:
                X.append((np.array(xs), np.array(sub_edges), np.array(sub_edge_features)))
                Y.append(np.array(sy))
                print("Found {} edges in segment {}".format(len(sub_edges), seg_no))

        pickle.dump([X, Y], open(os.path.join(param_fdp_labels, pattern, "{}.npd".format(vid)), "wb"))

    if summary_clips:
        summary = []
        for k in range(NA2):
            wh = np.array(dataset[k])
            avg = np.average(wh, axis=0)
            std = np.std(wh, axis=0)
            summary.append([activities[k], len(dataset[k]), avg[0], avg[1], std[0], std[1]])

        with open('summary.csv', 'w') as myfile:
            wr = csv.writer(myfile)
            for r in summary:
                wr.writerow(r)

    print("Done")


def sprel(bi, bj):
    """
    Spatial relevance of two objects defined by their bounding boxes
    :param bi:
    :param bj:
    :return:
    """
    xx1 = np.maximum(bi[0], bj[0])
    yy1 = np.maximum(bi[1], bj[1])
    xx2 = np.minimum(bi[2], bj[2])
    yy2 = np.minimum(bi[3], bj[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h
    if intersection > 0:
        area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
        area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
        return 0.5 * (1. + intersection / np.minimum(area_i, area_j))
    else:
        side_i = np.minimum((bi[2] - bi[0]), (bi[3] - bi[1]))
        side_j = np.minimum((bj[2] - bj[0]), (bj[3] - bj[1]))
        dist = np.minimum(math.fabs(xx2 - xx1), math.fabs(yy2 - yy1)) \
            if (xx2 - xx1) * (yy2 - yy1) < 0 \
            else math.sqrt((xx2 - xx1)**2 + (yy2 - yy1)**2)
        return 0.5 * (side_i + side_j) / (side_i + side_j + dist)


if __name__ == '__main__':

    # a1, only use a1 features
    # a2, only use a2 features
    # all, use a1 and a2 features

    # pattern = "a1"
    # prepare_data(pattern)

    # pattern = "a2"
    # prepare_data(pattern)

    pattern = "all"
    prepare_data(pattern, summary_clips=True)


















