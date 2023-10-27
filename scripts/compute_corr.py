# Standard Library
import os
import argparse
import itertools
import re
import json
import csv

# Public Library
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

# Custom
from const import REGISTERED_COMBINATIONS
from metrics import procrustes_sim, lin_cka_sim, pwcca_sim, error_dis, j_div

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

def load_data(tp, obj_model, ref_models, dataset, ind, tm):

    pred_obj_, pred_ref_ = None, None
    # Which layer to extract from (here the pooling)
    layer = 'x_flat' 
    files = os.listdir(os.path.join('..', 'pred_sets', dataset))

    # For the objective mode
    print("Loading {} predictions for objective model".format(tp))
    # If train data, get the feature/prediction for the similarity 
    if tp not in ['fuzz', 'gen']:
       obj_path = os.path.join('..', 'pred_sets', dataset, '{}_{}.npz'.format(obj_model + '_' + str(ind), tp))
       if not os.path.exists(obj_path):
        raise Exception("Pred set {} does not exist".format(obj_path))
       else:
        obj_path = os.path.join('..', 'pred_sets', dataset, '{}_{}.npz'.format(obj_model + '_' + str(ind), tp))
        obj = np.load(obj_path)
        pred_obj = obj[layer].copy()
        pred_obj_ = obj['pred'].copy()
        del obj
    # Else, just need the prediction
    else:
       obj_path = os.path.join('..', 'pred_sets', dataset, '{}_{}.npz'.format(obj_model + '_' + str(ind), tp))
       ref_obj = [[os.path.join('..', 'pred_sets', dataset, f) for f in files if re.match(re.escape(mod)+'_[0-9]{1}' \
        + re.escape('_{}_{}.npz'.format(tp, tm)), f) is not None] for mod in ref_models]
       pred_obj = []
       pred_obj_ = []
       for p_list in ref_obj:
         for p in p_list:
          if not os.path.exists(p):
           raise Exception("Pred set {} does not exist".format(p))
          else:
            loaded = np.load(p)
            pred_obj.append(loaded['pred'].copy())
            del loaded

    # For the reference models
    print("Loading {} predictions for reference models".format(tp))
    ref_path = [[os.path.join('..', 'pred_sets', dataset, f) for f in files if re.match(re.escape(mod)+'_[0-9]{1}' \
        + re.escape('_{}.npz'.format(tp)), f) is not None] for mod in ref_models]
    pred_ref, pred_ref_ = [], []
    nb_mods = []
    models_list = []
    for p_list, mod in zip(ref_path, ref_models):
      for p in p_list:
        models_list.append(p)
        if not os.path.exists(p):
           raise Exception("Pred set {} does not exist".format(p))
        else:
            if obj_path == p:
                continue
            if tp in ['train', 'test']:
                loaded = np.load(p)
                temp = loaded[layer].copy()
                pred_ref.append(temp) 
                pred_ref_.append(loaded['pred'].copy()) 
                del loaded
            else:
                l = np.load(p)
                pred_ref.append(l['pred'].copy())
                del l
      nb_mods.extend([(len(p_list), mod)]*len(p_list))

    return pred_obj, pred_obj_, pred_ref, pred_ref_, nb_mods, models_list

def corr_plot(pred_ref, div, closeness, nb_mods, sim_n, models_list):
    my_index = []
    for i in range(len(pred_ref)):
      # get the misclassifications
      my_index.append(np.where(pred_ref[i] == False)[0])
    
    a2, a3, = [], []
    save_a = []
    vals, lab = np.unique([nb_mods[k][1] for k in range(len(my_index))], return_inverse=True)

    for k in range(len(my_index)):
        a2.append(closeness[1][k]) 
        if div is not None:
            a3.append(len(set(div['list_covered_clusters_ref_'+str(nb_mods[k][1])+'_'+str(k % nb_mods[k][0])]) & set(div['list_covered_clusters_obj']))/len(div['list_covered_clusters_obj']))
            save_a.append(list(set(div['list_covered_clusters_ref_'+str(nb_mods[k][1])+'_'+str(k % nb_mods[k][0])]) & set(div['list_covered_clusters_obj'])))
        else:
            raise Exception("Should have provided the fault type coverage dict")
        
    print("Correlation Incorrect test input diversity transferability")
    sim_ = kendalltau(a3, a2)
    print(sim_)

    fig = plt.figure(figsize=(15, 7.5))    
    for i in range(len(vals)):
        plt.scatter(np.array(a2)[np.where(lab == i)[0]], np.array(a3)[np.where(lab == i)[0]], label=vals[i])   
    
    if sim_n not in ['acc', 'err', 'j_div']:
       plt.scatter(np.array(a2)[closeness[0][-5:]], np.array(a3)[closeness[0][-5:]], edgecolors='black', facecolors='none')
       plt.scatter(np.array(a2)[closeness[0][-1]], np.array(a3)[closeness[0][-1]], edgecolors='red', facecolors='none')
       
       # Top-1
       top_1_diversity = (np.array(a3)[closeness[0][:-1]] > np.array(a3)[closeness[0][-1]]).sum()/len(closeness[0])
       val_1_diversity = np.array(a3)[closeness[0][-1]]

       # Mean Top-5
       top_mean_5_diversity = top_1_diversity
       val_mean_5_diversity = val_1_diversity

       for k in range(1, 5):
          top_mean_5_diversity += (np.array(a3)[closeness[0][:-(k+1)]] > np.array(a3)[closeness[0][-(k+1)]]).sum()/(len(closeness[0][:-(k+1)])+1)
          val_mean_5_diversity += np.array(a3)[closeness[0][-(k+1)]]

       top_mean_5_diversity /= 5
       val_mean_5_diversity /= 5

       # Combining best
       at_most = 4
       best_list, first_list = [], []
       for ll in range(2, at_most + 1):       
           best = closeness[0][-ll:]
           print("Models: ", models_list[closeness[0][-ll:]][::-1])
           best_worst = set()
           for k in range(len(best)):
               best_worst = best_worst.union(set(save_a[best[k]]))
           
           print("Best of {} diversity: {}".format(ll, len(best_worst)/div['nb_covered_clusters_obj']))

           first = sorted([[val == models_list[closeness[0][::-1]][r].split('/')[-1].split('_')[0] for r in range(len(closeness[0]))].index(True) for val in vals])[:ll]
           print("Models: ", models_list[closeness[0][::-1]][first])
           first_worst = set()
           for k in range(len(first)):
               first_worst = first_worst.union(set(save_a[closeness[0][::-1][first[k]]]))
                      
           print("First of each model, diversity: ", len(first_worst)/div['nb_covered_clusters_obj'])

                    
           best_list.append(len(best_worst)/div['nb_covered_clusters_obj'])
           first_list.append(len(first_worst)/div['nb_covered_clusters_obj'])          
       
    else:
       plt.scatter(np.array(a2)[closeness[0][:5]], np.array(a3)[closeness[0][:5]], edgecolors='black', facecolors='none')
       plt.scatter(np.array(a2)[closeness[0][0]], np.array(a3)[closeness[0][0]], edgecolors='red', facecolors='none')
       
       # Top-1
       top_1_diversity = (np.array(a3)[closeness[0][1:]] > np.array(a3)[closeness[0][0]]).sum()/len(closeness[0])
       val_1_diversity = np.array(a3)[closeness[0][0]]

       # Mean Top-5
       top_mean_5_diversity = top_1_diversity
       val_mean_5_diversity = val_1_diversity
       for k in range(1, 5):
          top_mean_5_diversity += (np.array(a3)[closeness[0][(k+1):]] > np.array(a3)[closeness[0][k]]).sum()/(len(closeness[0][(k+1):])+1)
          val_mean_5_diversity += np.array(a3)[closeness[0][(k+1)]]

       top_mean_5_diversity /= 5
       val_mean_5_diversity /= 5

       # Combining best
       at_most = 4
       best_list, first_list = [], []
       for ll in range(2, at_most + 1):
           best = closeness[0][:ll]
           print("Models: ", models_list[closeness[0][:ll]])
           best_worst = set()
           for k in range(len(best)):
               best_worst = best_worst.union(set(save_a[best[k]]))
           print("Best of {} diversity: {}".format(ll, len(best_worst)/div['nb_covered_clusters_obj']))

           first = sorted([[val == models_list[closeness[0]][r].split('/')[-1].split('_')[0] for r in range(len(closeness[0]))].index(True) for val in vals])[:ll]
           print("Models: ", models_list[closeness[0]][first])
           first_worst = set()
           for k in range(len(first)):
               first_worst = first_worst.union(set(save_a[closeness[0][first[k]]]))
           print("First of each model, diversity: ", len(first_worst)/div['nb_covered_clusters_obj'])
           
           best_list.append(len(best_worst)/div['nb_covered_clusters_obj'])
           first_list.append(len(first_worst)/div['nb_covered_clusters_obj'])

    # / len(closeness[0]) for comparison between top-1 and top-5 (otherwise, top-5 might be higher than top-1 just because different number of models)
    print("Top-1 similarity for fault type coverage: {:.3f}, Top-mean_5 similarity for fault type coverage: {:.3f}".format(top_1_diversity, top_mean_5_diversity))
    plt.title("Similarity {} with fault type coverage: ({:.2f}, {:.2e})".format(sim_n, sim_.correlation, sim_.pvalue))
    plt.legend()
        
    return sim_, top_1_diversity, top_mean_5_diversity, val_1_diversity, val_mean_5_diversity, \
     sim_n, first_list, best_list

# Following https://github.com/js-d/sim_metric/blob/main/dists/score_pair.py
def normalize(rep1, rep2):
    # center each row
    rep1 = rep1 - rep1.mean(axis=0, keepdims=True)
    rep2 = rep2 - rep2.mean(axis=0, keepdims=True)

    # normalize each representation
    rep1 = rep1 / np.linalg.norm(rep1)
    rep2 = rep2 / np.linalg.norm(rep2)

    return rep1.T, rep2.T

def get_sim_rank(obj_list, ref_list, sim_type='acc'):
    dist_tr = []
    if sim_type == 'pwcca':
        for (obj, ref) in zip(obj_list, ref_list):
            obj, ref = normalize(obj, ref)
            sim = pwcca_sim(obj, ref)
            dist_tr.append(sim)
    elif sim_type == 'cka':
        for (obj, ref) in zip(obj_list, ref_list):
            obj, ref = normalize(obj, ref)
            dist_tr.append(lin_cka_sim(obj, ref))
    elif sim_type == 'ortho':
        for (obj, ref) in zip(obj_list, ref_list):
            obj, ref = normalize(obj, ref)
            dist_tr.append(procrustes_sim(obj, ref))
    elif sim_type == 'acc':
        for (obj, ref) in zip(obj_list, ref_list):
            obj = lambda_max(obj, axis=1, key=np.abs) >= 0
            ref = lambda_max(ref, axis=1, key=np.abs) >= 0
            dist_tr.append(np.abs(np.sum(obj)/len(obj) - np.sum(ref)/len(ref)))
    elif sim_type == 'err':
        for (obj, ref) in zip(obj_list, ref_list):
            #obj_ = np.sum(lambda_max(obj, axis=1, key=np.abs) >= 0)/len(obj)
            #ref_ = np.sum(lambda_max(ref, axis=1, key=np.abs) >= 0)/len(ref)
            dist_tr.append(error_dis(obj, ref))
    elif sim_type == 'j_div':
        for (obj, ref) in zip(obj_list, ref_list):
            dist_tr.append(j_div(obj, ref))

    return dist_tr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ref_model', help='List of reference models to use. By default, will use all models except eventually the objective model.', nargs='+', default=None)
    parser.add_argument('-o', '--obj_model', help='Objective model to compare to reference', default=None)
    parser.add_argument('-i', '--ind', help='Random instance among objective model to use', default=0, type=int)
    parser.add_argument('-d', '--dataset', help='Dataset to use for computation', default='cifar10')
    parser.add_argument('-t', '--type', help='Test set type (fuzz, gen)', default=None)
    parser.add_argument('--save', help='save in a .csv file the results', action='store_true')
    parser.add_argument('--plot', help='whether or not to plot the results', action='store_true')
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

    if len(models_choice) == 0:
       raise Exception("Empty reference models list, meaning only one reference model was provided and the model is the same as the objective model.")

    print("Similarity of {} against {}".format(args.obj_model + '_' + str(args.ind), models_choice))
    pred_obj_tr, acc_obj, pred_ref_tr, acc_ref, _, _ = load_data('train', args.obj_model, models_choice, args.dataset, args.ind, str(args.obj_model) + '_' + str(args.ind))
    
    with open(os.path.join('..', 'pred_sets', args.dataset, 'cov_type2_cluster_{}_{}_{}.json'.format(args.obj_model, args.ind, args.type)), "r") as f:
        div_ts = json.load(f)
    
    _, _, pred_ref_ts, _, nb_mods, models_list = load_data(args.type, args.obj_model, models_choice, args.dataset, args.ind, str(args.obj_model) + '_' + str(args.ind))
        
    sim_dict = {}
    sim_list =['pwcca', 'cka', 'ortho', 'acc', 'err', 'j_div']
    for args.sim in sim_list:
        print("Calculating similarity {}".format(args.sim))
        dist_tr = get_sim_rank(itertools.repeat(pred_obj_tr) if args.sim in ['pwcca', 'cka', 'ortho'] else itertools.repeat(acc_obj),
                               pred_ref_tr if args.sim in ['pwcca', 'cka', 'ortho'] else acc_ref,
                               args.sim)

        closeness = ( np.argsort(dist_tr), dist_tr)
        sim_dict[args.sim] = closeness
                
    for i in range(len(sim_list)):
        print("Calculating for : ", sim_list[i])
        sim_, top_1_diversity, top_mean_5_diversity, val_1_diversity, val_mean_5_diversity,\
            sim, overall_first_worst, overall_best_worst = corr_plot([lambda_max(n, axis=1, key=np.abs) >= 0 for n in pred_ref_ts], \
                                                                  div_ts, sim_dict[sim_list[i]], nb_mods, sim_list[i], np.array(models_list))
        if args.plot:
           print("Plotting for : ", sim_list[i])
           plt.show()
        if args.save:
            with open('res.csv', 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([args.obj_model, sim_[0], sim_[1], top_1_diversity, top_mean_5_diversity, val_1_diversity, val_mean_5_diversity, \
                    sim[0], sim[1]])
            with open('res_mult.csv', 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow([args.obj_model, overall_first_worst[0], overall_first_worst[1], overall_first_worst[2],\
                overall_best_worst[0], overall_best_worst[1], overall_best_worst[2]])

