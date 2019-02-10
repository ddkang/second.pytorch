import argparse
import json
import os
import feather
import numpy as np
import pandas as pd
import scipy.stats
from collections import defaultdict


NB_TAKE = 100 

def write_out(round_ind, rep, strat, data_name, img_inds):
    dir_name = './round{}/rep{}/{}/'.format(round_ind + 1, rep, strat)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fname = os.path.join(dir_name, '{}.txt'.format(data_name))

    img_inds.sort()
    with open(fname, 'w') as f:
        for ind in img_inds:
            f.write('%06d' % ind)
            f.write('\n')

def load_inds(rnd, rep, strat_name, data_name):
    dir_name = './round{}/rep{}/{}/'.format(rnd, rep, strat_name)
    fname = os.path.join(dir_name, '{}.txt'.format(data_name))
    inds = open(fname, 'r').readlines()
    inds = map(lambda x: int(x.strip()), inds)
    return list(inds)

def load_lidar_preds(rnd, rep, strat_name):
    tmp = '/mnt/disks/data/kitti/data/models/round{}/rep{}/{}/eval_results/'\
        .format(rnd, rep, strat_name)
    tmp2 = os.listdir(tmp)[0]
    preds_dir = os.path.join(tmp, tmp2)

    boxes = []
    for fname in os.listdir(preds_dir):
        im_ind = int(fname[0:6])
        fname = os.path.join(preds_dir, fname)
        lines = open(fname).readlines()
        lines = map(lambda s: s.strip(), lines)
        lines = map(lambda s: s.split(' '), lines)
        lines = list(lines)
        if len(lines) == 0:
            continue

        for line in lines:
            line[1:] = [float(s) for s in line[1:]]
            if line[0] != 'Car':
                print(line[0])
            else:
                line[0] = 3
            box = [im_ind, line[0], line[-1], line[4], line[5], line[6], line[7]]
            boxes.append(box)
    df = pd.DataFrame(
        boxes,
        columns=['im_ind', 'cname', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'])
    f32 = ['conf', 'xmin', 'ymin', 'xmax', 'ymax']
    for f in f32:
        df[f] = df[f].astype('float32')
    return df

def load_mask_preds():
    df = feather.read_dataframe('./train-obj.feather')
    df['im_ind'] = df['frame'].map(lambda x: int(x.split('/')[-1][:-4]))
    return df

def box_iou(box1, box2):
    def to_cent(box):
        xmin, ymin, xmax, ymax = box
        xcent = (xmax + xmin) / 2
        ycent = (ymax + ymin) / 2
        return (xcent, ycent, xmax - xcent, ymax - ycent)
    box1 = to_cent(box1)
    box2 = to_cent(box2)
    if box1[0] + box1[2] <= box2[0] - box2[2] or \
            box2[0] + box2[2] <= box1[0] - box1[2] or \
            box1[1] + box1[3] <= box2[1] - box2[3] or \
            box2[1] + box2[3] <= box1[1] - box1[3]:
        return 0.0
    else:
        xA = min(box1[0] + box1[2], box2[0] + box2[2])
        yA = min(box1[1] + box1[3], box2[1] + box2[3])
        xB = max(box1[0] - box1[2], box2[0] - box2[2])
        yB = max(box1[1] - box1[3], box2[1] - box2[3])
        interArea = (xA - xB) * (yA - yB)
        box1Area = (2 * box1[2]) * (2 * box1[3])
        box2Area = (2 * box2[2]) * (2 * box2[3])
        return max(interArea / float(box1Area + box2Area - interArea), 0.0)

def get_groups(df):
    groups = defaultdict(list)
    for row in df.itertuples():
        groups[row.im_ind].append(row)
    return groups

def count_invs(lidar_preds, mask_preds, inds):
    invs = []
    def overlaps(box1, box2, cutoff=0.5):
        def get_box(tmp):
            return (tmp.xmin, tmp.ymin, tmp.xmax, tmp.ymax)
        iou = box_iou(get_box(box1), get_box(box2))
        return iou > cutoff
    def count(boxes1, boxes2):
        b1 = []
        b2 = []
        for box1 in boxes1:
            flag = True
            for box2 in boxes2:
                if overlaps(box1, box2):
                    flag = False
                    break
            if flag:
                b1.append(box1)
        for box2 in boxes2:
            flag = True
            for box1 in boxes1:
                if overlaps(box1, box2):
                    flag = False
                    break
            if flag:
                b2.append(box2)
        return len(b1) + len(b2)

    glidar, gmask = get_groups(lidar_preds), get_groups(mask_preds)
    for ind in inds:
        invs.append(count(glidar[ind], gmask[ind]))
    return invs


def inds_to_split(orig_train, orig_unlabeled, inds):
    new_train = []
    for datum in orig_train:
        new_train.append(datum)
    for ind in inds[:NB_TAKE]:
        new_train.append(orig_unlabeled[ind])

    new_unlabeled = []
    for ind in inds[NB_TAKE:]:
        new_unlabeled.append(orig_unlabeled[ind])
    return new_train, new_unlabeled


def get_rand(orig_train, orig_unlabeled, lidar_preds, mask_preds):
    inds = list(range(len(orig_unlabeled)))
    np.random.seed(1)
    np.random.shuffle(inds)

    return inds_to_split(orig_train, orig_unlabeled, inds)


def get_omg(orig_train, orig_unlabeled, lidar_preds, mask_preds):
    nb_invs = count_invs(lidar_preds, mask_preds, orig_unlabeled)
    assert len(nb_invs) == len(orig_unlabeled)

    tmp = zip(nb_invs, orig_unlabeled)
    tmp = list(tmp)
    tmp.sort(reverse=True)
    _, unlabeled = zip(*tmp)

    inds = list(range(NB_TAKE // 2, len(orig_unlabeled)))
    np.random.seed(1)
    np.random.shuffle(inds)
    inds = list(range(0, NB_TAKE // 2)) + inds

    return inds_to_split(orig_train, unlabeled, inds)


def get_omg_all(orig_train, orig_unlabeled, lidar_preds, mask_preds):
    nb_invs = count_invs(lidar_preds, mask_preds, orig_unlabeled)
    assert len(nb_invs) == len(orig_unlabeled)

    tmp = zip(nb_invs, orig_unlabeled)
    tmp = list(tmp)
    tmp.sort(reverse=True)
    _, unlabeled = zip(*tmp)

    inds = list(range(len(orig_unlabeled)))
    return inds_to_split(orig_train, unlabeled, inds)


def get_active_learning(orig_train, orig_unlabeled, lidar_preds, callback):
    # probs[i][0] is the nb_inv
    def get_probs():
        ret = []
        for ind in orig_unlabeled:
            tmp = []
            # FIXME
            df = lidar_preds[lidar_preds['im_ind'] == ind]
            for pred in df.itertuples():
                tmp.append(pred.conf)
            ret.append(tmp)
        return ret
    probs = get_probs()

    gap = list(map(callback, probs))
    tmp = zip(gap, orig_unlabeled)
    tmp = list(tmp)
    tmp.sort(reverse=True)
    _, unlabeled = zip(*tmp)

    inds = list(range(len(unlabeled)))

    return inds_to_split(orig_train, unlabeled, inds)


def get_least_confident(orig_train, orig_unlabeled, lidar_preds, mask_preds):
    def compute_gap(prob):
        if len(prob) == 0:
            return 0
        max_pred = max(prob)
        return 1 - max_pred
    return get_active_learning(orig_train, orig_unlabeled, lidar_preds, compute_gap)

def get_max_margin(orig_train, orig_unlabeled, lidar_preds, mask_preds):
    def compute_margin(prob):
        # NOTE: currently for car only, so this works
        if len(prob) == 0:
            return 0
        tmp = map(lambda x: abs(1 - 2 * x), prob)
        tmp = sum(tmp) / len(prob)
        return tmp
    return get_active_learning(orig_train, orig_unlabeled, lidar_preds, compute_margin)

def get_max_entropy(orig_train, orig_unlabeled, lidar_preds, mask_preds):
    def compute_entropy(prob):
        if len(prob) == 0:
            return 0
        tmp = map(lambda x: scipy.stats.entropy([x, 1-x]), prob)
        tmp = sum(tmp) / len(prob)
        return tmp
    return get_active_learning(orig_train, orig_unlabeled, lidar_preds, compute_entropy)
    

def main():
    parser = argparse.ArgumentParser()
    # NOTE: takes round i, outputs to round i+1
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--rep', type=int, required=True) # replicate
    args = parser.parse_args()

    # Rand
    strats = [('rand', get_rand),
              ('omg', get_omg),
              ('omg-all', get_omg_all),
              ('least', get_least_confident),
              ('margin', get_max_margin),
              ('entropy', get_max_entropy)]
    # strats = [('omg-all', get_omg_all)]

    for strat_name, callback in strats:
        orig_train = load_inds(args.round, args.rep, strat_name, 'train')
        orig_unlabeled = load_inds(args.round, args.rep, strat_name, 'unlabeled')
        all_lidar_preds = load_lidar_preds(args.round, args.rep, strat_name)
        all_mask_preds = load_mask_preds()

        new_train, new_unlabeled = \
            callback(orig_train, orig_unlabeled, all_lidar_preds, all_mask_preds)
        write_out(args.round, args.rep, strat_name, 'train', new_train)
        write_out(args.round, args.rep, strat_name, 'unlabeled', new_unlabeled)
        # TODO: symlink(?) val, etc cause the datagen script is dumb

if __name__ == '__main__':
    main()
