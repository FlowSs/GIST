# Standard library
import os
import argparse
import re
import json
from operator import itemgetter

import warnings
warnings.filterwarnings("ignore")

# Public library
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# Custom
from const import REGISTERED_COMBINATIONS


def lambda_max(arr, axis=None, key=None, keepdims=False):
    if callable(key):
        idxs = np.argmax(key(arr), axis)
        if axis is not None:
            idxs = np.expand_dims(idxs, axis)
            result = np.take_along_axis(arr, idxs, axis)
            if not keepdims:
                result = np.squeeze(result, axis=axis)
            return result
        else:
            return arr.flatten()[idxs]
    else:
        return np.amax(arr, axis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref_model', help='List of reference models to use. By default, will use all models except eventually the objective model.', nargs='+', default=None)
    parser.add_argument('-o', '--obj_model', help='Objective model to compare to reference', default=None)
    parser.add_argument('-i', '--ind', help='Random instance among objective model to use', default=None, type=int)
    parser.add_argument('-d', '--dataset', help='Dataset to use for computation', default='cifar10')
    parser.add_argument('-t', '--type', help='Test set type (test, fuzz, adv, gen)', default=None)
    args = parser.parse_args()

    assert args.dataset in REGISTERED_COMBINATIONS, "Dataset {} not recognised".format(args.dataset)   
    assert args.obj_model in REGISTERED_COMBINATIONS[args.dataset], "Objective model {} not recognised".format(args.obj_model)
    
    if args.ref_model is None:
        args.ref_model = REGISTERED_COMBINATIONS[args.dataset]

    models_choice = []
    for mod in args.ref_model:
        assert mod in REGISTERED_COMBINATIONS[args.dataset], "Model {} not recognised for dataset {}".format(mod, args.dataset)
        if mod != args.obj_model:
            models_choice.append(mod)

    files = os.listdir(os.path.join('..', 'models', args.dataset))
    mod_lab = []

    with open(os.path.join('..', 'pred_sets', args.dataset, 'cov_type2_cluster_{}_{}_{}.json'.format(args.obj_model, args.ind, args.type)), "r") as f:
        div = json.load(f)
    clust_in_obj = div['list_covered_clusters_obj']
    ref_rep = []
    order_ref = []

    # load and add test set for the ref models
    for mods in models_choice:
        file_list = [f for f in files if re.match(re.escape(mods)+'_[0-9]{1}'+re.escape('.'), f) is not None]
                
        for f in file_list:
            if mods == args.obj_model:
                continue
                    
            i = int(f.split('_')[1].split('.')[0])
                    
            print("Using test set of type {} from model {}".format(args.type, str(mods)+'_'+str(i)))
            clust_in_ref, nb_in_ref = np.unique(div['cluster_per_seed_'+mods+'_'+str(i)], return_counts=True)
            ref_rep.append(np.zeros(shape=(len(clust_in_obj))))
            # For each of reference test sets, we check if it 
            # cover a fault type that the objective test set
            # did and put the number of inputs that did so
            for (c, n) in zip(clust_in_ref, nb_in_ref):
                if c != -1 and c in clust_in_obj:
                    ref_rep[-1][clust_in_obj.index(c)] = n

            order_ref.append(mods+'_'+str(i))
    ref_rep = np.array(ref_rep)

    # Hierarchical clustering
    Z = linkage(ref_rep, 'ward')

    # empirically choosing value to avoid too many clusters
    # as well as cutting dendrogram roughly when the distance
    # between branching become large
    # just in order to have more interpretable results
    if args.dataset == 'cifar10':
        if args.type == 'fuzz':
            c = 0.8
        else:
            c = 0.9
    else:
        c = 0.6
    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Helvetica"
            })
    dn = dendrogram(Z, labels=order_ref, color_threshold=c*max(Z[:,2]))
    
    temp = [list(x) for x in zip(*sorted(zip(dn['ivl'], dn['leaves_color_list']), key=itemgetter(0)))]
    obtained_labels, cluster_list = temp[0], temp[1]
    unique_obtained_labels = np.unique([l.split('_')[0] for l in obtained_labels]).tolist()

    ax = plt.gca()
    ax.set_yticklabels(ax.get_yticklabels(), size=20)
    ax.set_xticklabels(ax.get_xticklabels(), size=15,ha="right")
    plt.tight_layout()
    plt.show()
    

