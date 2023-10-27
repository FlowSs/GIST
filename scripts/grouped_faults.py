# Standard libraru
import os
import argparse
import re
import json

import warnings
warnings.filterwarnings("ignore")

# Public library
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import skimage.measure

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
    parser.add_argument('-t', '--type', help='Test set type (test, fuzz, adv, gen)', default='normal')
    parser.add_argument('--heatmap', help='to display the final heatmap or not', action='store_true')
    args = parser.parse_args()

    assert args.dataset in REGISTERED_COMBINATIONS, "Dataset {} not recognised".format(args.dataset)   
    
    # If heatmap argument provided, display it
    # Otherwise, will create and/or populate the matrix
    # to do so
    if args.heatmap:  
        sns.set(font_scale=1.5, rc={'text.usetex' : True})  
        tmp = np.load(os.path.join('..', 'data', args.dataset, 'faults_covering_{}.npz'.format(args.type)))
        matrix = tmp['mat']    
        tmp_lab = REGISTERED_COMBINATIONS[args.dataset]
        matrix = skimage.measure.block_reduce(matrix, (10,10), np.mean)
        sns.heatmap(matrix, xticklabels=tmp_lab, yticklabels=tmp_lab, cmap='Reds', annot=True, fmt='.3f', mask=np.diag(np.ones(5)))
        if args.dataset == 'mr':
            plt.yticks(rotation=0)
        if args.dataset == 'cifar10':
            plt.xticks(rotation=0)
        plt.xlabel('Reference models', fontsize=15)
        plt.ylabel('Objective models', fontsize=15)
        plt.tight_layout()
        plt.show()
    else:
        assert args.obj_model in REGISTERED_COMBINATIONS[args.dataset], "Objective model {} not recognised".format(args.obj_model)

        if args.ref_model is None:
            args.ref_model = REGISTERED_COMBINATIONS[args.dataset]

        models_choice = []
        for mod in args.ref_model:
            assert mod in REGISTERED_COMBINATIONS[args.dataset], "Model {} not recognised for dataset {}".format(mod, args.dataset)
            models_choice.append(mod)

        files = os.listdir(os.path.join('..', 'models', args.dataset))
        mod_lab = []

        with open(os.path.join('..', 'pred_sets', args.dataset, 'cov_type2_cluster_{}_{}_{}.json'.format(args.obj_model, args.ind, args.type)), "r") as f:
            div = json.load(f)
        clust_in_obj = set(div['list_covered_clusters_obj'])
        ref_rep = np.zeros(shape=(50,))
        order_ref = []

        # load and add test set for the ref models
        for mods in models_choice:
                file_list = [f for f in files if re.match(re.escape(mods)+'_[0-9]{1}'+re.escape('.'), f) is not None]
                for f in file_list:
                    i = int(f.split('_')[1].split('.')[0])
                    order_ref.append(mods+'_'+str(i))

                    if mods == args.obj_model:
                        continue                
                    
                    print("Using test set of type {} from model {}".format(args.type, str(mods)+'_'+str(i)))
                    clust_in_ref = set(div['list_covered_clusters_ref_'+mods+'_'+str(i)])
                    ref_rep[len(order_ref)-1] = len(clust_in_ref & clust_in_obj)/len(clust_in_obj)
        
        if not(os.path.exists(os.path.join('..', 'data', args.dataset, 'faults_covering_{}.npz'.format(args.type)))):
            matrix = np.zeros(shape=(50, 50))
        else:
            matrix = np.load(os.path.join('..', 'data', args.dataset, 'faults_covering_{}.npz'.format(args.type)))
            assert (np.array(order_ref) == matrix['lab']).all()
            matrix = matrix['mat']

        i, j = REGISTERED_COMBINATIONS[args.dataset].index(args.obj_model), int(args.ind)
        matrix[i * 10 + j] = ref_rep
        np.savez(os.path.join('..', 'data', args.dataset, 'faults_covering_{}.npz'.format(args.type)), \
                mat=matrix, lab=np.array(order_ref))

