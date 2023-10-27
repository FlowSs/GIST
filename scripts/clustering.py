# Standard Library
import os
import argparse
import time
import re
import random
import warnings
warnings.filterwarnings("ignore")
import json

# Public library
import numpy as np
import pickle
import hdbscan
from hdbscan.validity import validity_index
import sklearn
import umap.umap_ as umap
from umap.umap_ import nearest_neighbors

# Own functions
from const import REGISTERED_COMBINATIONS

# Setting the seed for randomness
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# Return the label prediction based of a vector of softmax
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
    parser.add_argument('-i', '--ind', help='Random instance among objective model to use', default=0, type=int)
    parser.add_argument('-d', '--dataset', help='Dataset to use for computation', default='cifar10')
    parser.add_argument('-t', '--type', help='Test set type (fuzz or gen)', default=None)
    parser.add_argument('--save_mod', help='save models', action='store_true')
    parser.add_argument('--save_clust', help='save data from clusters. Only for calculating accuracy', action='store_true')
    args = parser.parse_args()
 
    set_seed()
    assert args.dataset in REGISTERED_COMBINATIONS, "Dataset {} not recognised".format(args.dataset)   
    assert args.obj_model in REGISTERED_COMBINATIONS[args.dataset], "Objective model {} not recognised".format(args.obj_model)

    if args.ref_model is None:
        args.ref_model = REGISTERED_COMBINATIONS[args.dataset]

    models_choice = []
    for mod in args.ref_model:
      assert mod in REGISTERED_COMBINATIONS[args.dataset], "Model {} not recognised for dataset {}".format(mod, args.dataset)
      if mod != args.obj_model:
         models_choice.append(mod)

    # test sets feature data
    global_test_data = []
    # predicted labels of the data
    global_label_data = []
    # ground truth of the data
    global_ground_truth = []
    # test sets data (raw inputs), just to have them saved when retraining the model with some data from the clusters
    global_orig_data = []

    # Consider the pooling layer
    layer_considered = 'x_flat'
    # which model the data belongs to
    # match global_test_data shape
    mod_lab = []

    if args.type == 'fuzz':
        test_data = np.load(os.path.join('..', 'pred_sets', args.dataset, '{}_{}_fuzz.npz'.format(args.obj_model, args.ind)), allow_pickle=True)
        global_test_data.append(test_data[layer_considered])
        tmp = np.load(os.path.join('..', 'data', args.dataset, 'fuzz_sets', '{}_{}_fuzz_results.npz'.format(args.obj_model, args.ind)))
        test_labels = tmp['labels']
        orig_data = tmp['advs']
        
        global_label_data.append(np.argmin(test_data['pred'], 1))
        global_ground_truth.append(test_labels)
        global_orig_data.append(orig_data)
        mod_lab.extend(['obj'] * len(test_labels))
    elif args.type == 'gen':
        if args.dataset == 'cifar10':
            test_data = np.load(os.path.join('..', 'pred_sets', args.dataset, '{}_{}_gen.npz'.format(args.obj_model, args.ind)), allow_pickle=True)
            global_test_data.append(test_data[layer_considered])
            tmp = np.load(os.path.join('..', 'data', args.dataset, 'gan_sets', '{}_{}_gen_results.npz'.format(args.obj_model, args.ind)))
            # Hardcoded: for GAN based, we have 100 data points for each of the 10 classes
            test_labels = np.concatenate([[i]*100 for i in range(10)])
            orig_data = tmp['advs']
            
            global_label_data.append(np.argmin(test_data['pred'], 1))
            global_ground_truth.append(test_labels)
            global_orig_data.append(orig_data)
            mod_lab.extend(['obj'] * len(test_labels))
        elif args.dataset == 'mr':
            test_data = np.load(os.path.join('..', 'pred_sets', args.dataset, '{}_{}_gen.npz'.format(args.obj_model, args.ind)), allow_pickle=True)
            global_test_data.append(test_data[layer_considered])
            tmp = np.load(os.path.join('..', 'data', args.dataset, 'gen', 'log_' + args.obj_model + '_' + str(args.ind) + '.npz'))
            test_labels = tmp['label']
            orig_data = tmp['text']
            
            global_label_data.append(np.argmin(test_data['pred'], 1))
            global_ground_truth.append(test_labels)
            global_orig_data.append(orig_data)
            mod_lab.extend(['obj'] * len(test_labels))
        else:
            raise Exception("Dataset of type {} not recognised for test set type 'gen'".format(args.dataset))

    else:
        raise Exception("Type of test set {} not recognised".format(args.type))

    files = os.listdir(os.path.join('..', 'models', args.dataset))

    # load and add test set for the ref models
    for mods in models_choice:
        file_list = [f for f in files if re.match(re.escape(mods)+'_[0-9]{1}'+re.escape('.'), f) is not None]
        
        for f in file_list:
            i = int(f.split('_')[1].split('.')[0])
            # Shouldn't happen that some models are of the same type of the objective model (because we excluded them)
            # but just in case
            if mods == args.obj_model:
              continue
            start_time = time.time()
            print("Using test set of type {} from model {}".format(args.type, str(mods)+'_'+str(i)))
            # load test set according to its definition
            if args.type == 'fuzz':
                test_data = np.load(os.path.join('..', 'pred_sets', args.dataset, '{}_{}_fuzz_{}_{}.npz'.format(mods, str(i), \
                                    args.obj_model, args.ind)))                
                pred_data = test_data['pred']
                feat_data = test_data[layer_considered]
                tmp = np.load(os.path.join('..', 'data', args.dataset, 'fuzz_sets', '{}_{}_fuzz_results.npz'.format(mods, str(i))))
                test_labels = tmp['labels']
                orig_data = tmp['advs']
                # Getting mispredicted inputs
                misc_inp_norm = np.where(lambda_max(test_data['pred'], axis=1, key=np.abs) < 0)[0]
                
                global_test_data.append(feat_data[misc_inp_norm])
                global_label_data.append(np.argmin(test_data['pred'][misc_inp_norm], 1))
                global_ground_truth.append(test_labels[misc_inp_norm])
                global_orig_data.append(orig_data[misc_inp_norm])

                mod_lab.extend([mods+'_'+str(i)] * len(test_labels[misc_inp_norm]))
            elif args.type == 'gen':
                test_data = np.load(os.path.join('..', 'pred_sets', args.dataset, '{}_{}_gen_{}_{}.npz'.format(mods, str(i), args.obj_model, args.ind)), allow_pickle=True)                
                feat_data = test_data[layer_considered]
                if args.dataset == 'cifar10':
                    orig_data = np.load(os.path.join('..', 'data', args.dataset, 'gan_sets', '{}_{}_gen_results.npz'.format(mods, str(i))))['advs']
                    # Hardcoded: for GAN based, we have 100 data points for each of the 10 classes
                    test_labels = np.concatenate([[i]*100 for i in range(10)])
                elif args.dataset == 'mr':
                    tmp = np.load(os.path.join('..', 'data', args.dataset, 'gen', 'log_' + mods + '_' + str(i) + '.npz'))
                    test_labels = tmp['label']
                    orig_data = tmp['text']

                # Getting mispredicted inputs
                misc_inp_norm = np.where(lambda_max(test_data['pred'], axis=1, key=np.abs) < 0)[0]

                global_test_data.append(feat_data[misc_inp_norm])
                global_label_data.append(np.argmin(test_data['pred'][misc_inp_norm], 1))
                global_ground_truth.append(test_labels[misc_inp_norm])
                global_orig_data.append(orig_data[misc_inp_norm])

                mod_lab.extend([mods+'_'+str(i)] * len(test_labels[misc_inp_norm]))

            else:
                raise Exception("Type of test set {} not recognised".format(args.type))
    
    # 9 for cifar10, 1 for mr
    # used to normalize the labels (going from 0 - 9/1)
    num_label = 9 if args.dataset == 'cifar10' else 1

    
    # Data preparation
    features = np.vstack(global_test_data)
    if args.dataset == 'cifar10':
        global_orig_data = np.vstack(global_orig_data)
    else:
        global_orig_data = np.concatenate((global_orig_data), axis=0)
    ground_truth = np.hstack(global_ground_truth)
    label_data = np.hstack(global_label_data)
    print(features.shape, ground_truth.shape, label_data.shape, global_orig_data.shape)

    # Stacking features and normalizing them by column, following the other paper methodology
    X_features = features.reshape(len(features), np.prod(features.shape[1:]))
    X_min = X_features.min(axis=0)
    X_max = X_features.max(axis=0)
    X_features = (X_features - X_min)/(X_max - X_min + 1e-8)

    # Adding the normalized predicted/ground truth label as extra features as the original method
    TY = ground_truth/num_label 
    PY = label_data/num_label
    c= np.c_[X_features, TY]
    c_orig=np.c_[c, PY]
    print(c_orig.shape)

    # Dict with all values to save for the search
    best = {'shs': -1, 'dbcv': -1, 'noisy': 100, 'n_n': None, 'i': None, 'k': None, 'min_cluster_size': None, 'm_size': None}
    best_clusterer = None
    best_um = None

    # Parameters that leads to good DBCV/Silhouette score (hand picked)
    # We just will search over the number of components of UMAP
    # Noise level < 10 for Fuzz and Text method
    # Noise level < 15 for GAN, because of the generation method
    # which add extra noise (The GAN generated distribution is not exactly the same as the input distribution of the model)
    if args.type == 'fuzz':
        
        n_n = 0.0
        k = 3 
        m_size = 5
        m_samples = 5
        epsilon =  0.25
        n_comp_list = [150, 100, 50, 25, 10]
        noise_level = 10        
        
    elif args.type == 'gen':
        if args.dataset == 'cifar10':

            n_n = 0.0
            k = 3 if any(ele in args.obj_model for ele in ['vgg']) else 2
            m_size = 5
            m_samples = 5 if any(ele in args.obj_model for ele in ['vgg']) else 1
            epsilon = 0.15 if any(ele in args.obj_model for ele in ['vgg']) else 2
            n_comp_list = [150, 100, 50, 25, 10]
            noise_level = 15

        elif args.dataset == 'mr':
            
            n_n = 0.0
            k = 5
            m_size = 5 
            m_samples = 5
            epsilon = 0.25
            n_comp_list = [150, 100, 50, 25, 10]
            noise_level = 10            


    # Precompute nearest neighbors for the search
    print("Precomputing kNN for UMAP")
    precomputed_knn = nearest_neighbors(
                        c_orig,
                        n_neighbors=10,
                        metric_kwds=None,
                        angular=None,
                        metric='euclidean',
                        random_state=42,
                        )
    # Search
    # Fixing the random seed as much as possible for reproducibility
    # Can vary a bit depending on architecture but should be relatively the same if the data are the same
    for i in n_comp_list:
              if i >= X_features.shape[1]:
                 continue
              print("Searching for:")
              print("Nb of components: {} and Number of neighbors: {} and Min_dist: {} for UMAP".format(i, k, n_n) )
              print("Epsilon: {}, Min sample size: {}, Min cluster size: {} for HDBSCAN".format(epsilon, m_samples, m_size))
              
              # UMAP : Dimensionality Reduction
              um = umap.UMAP(min_dist=n_n, n_components=i, n_neighbors=k, random_state=42, precomputed_knn=precomputed_knn)            
              c_orig_ = um.fit_transform(c_orig)

              # HDBSCAN : CLustering
              clusterer1 = hdbscan.HDBSCAN(min_cluster_size = m_size, min_samples=m_samples, cluster_selection_epsilon=epsilon, prediction_data=True)
              clusterer1.fit(c_orig_)

              a, _ = np.unique(clusterer1.labels_, return_counts=True)
              total_nb_cluster_ = len(a) - 1
              #print(total_nb_cluster_)
              ret_ = clusterer1.labels_
                            
              l , cc = np.unique(ret_, return_counts=True)
              nb_covered_clusters_obj_ = len(np.unique(ret_[np.where(np.array(mod_lab) == 'obj')[0]]))
              if -1 in np.unique(ret_[np.where(np.array(mod_lab) == 'obj')[0]]):
                nb_covered_clusters_obj_ -= 1
              noisy_inputs_ = cc[0]
              
              # Silhouette score and DBCV implementation 
              shs_ = sklearn.metrics.silhouette_score(c_orig_, ret_)
              dbcv_s_ = validity_index(c_orig_.astype('double'), ret_)
              print("Number of clusters: {}".format(total_nb_cluster_))
              print("Min dist: {}, Number of components: {}, Number of neighbors: {}".format(n_n, i, k))
              print("Epsilon: {}, Min sample size: {}, Min cluster size: {}".format(epsilon, m_samples, m_size))
              print("silhouette_score: {}, DBCV score: {}, Percentage noisy inputs in test set: {:.2f}".format(shs_, dbcv_s_, noisy_inputs_/len(label_data)*100))
                
              # Maxmize DBCV score and make sure not too many noise
              # Also check for at least 80 clusters (since we don't want them to be too big i.e. too general)
              # We rely mainly on DBCV since we use HDBSCAN which is density based, since clusters won't necessarily be spherical which
              # can lead to Silhouette score being low and DBCV high. 
              # In practice, Silhouette score followed DBCV score
              if dbcv_s_ > best['dbcv'] and noisy_inputs_/len(label_data)*100 < noise_level and total_nb_cluster_ > 80:
                print("New best with following parameters:")
                print("Number of clusters: {}".format(total_nb_cluster_))
                print("Min dist: {}, Number of components: {}, Number of neighbors: {}".format(n_n, i, k))
                print("Epsilon: {}, Min sample size: {}, Min cluster size: {}".format(epsilon, m_samples, m_size))
                print("silhouette_score: {}, DBCV score: {}, Percentage noisy inputs in test set: {:.2f}".format(shs_, dbcv_s_, noisy_inputs_/len(label_data)*100))
                #print(cc)
                best['n_n'], best['i'], best['k'] = n_n, i, k
                best['m_samples'], best['m_size'], best['epsilon'] = m_samples, m_size, epsilon
                best['shs'], best['dbcv'], best['noisy'] = shs_, dbcv_s_, noisy_inputs_/len(label_data)*100
                best_clusterer = clusterer1
                best_um = um
                total_nb_cluster = total_nb_cluster_
                nb_covered_clusters_obj = nb_covered_clusters_obj_
                shs, dbcv_s, noisy_inputs = shs_, dbcv_s_, noisy_inputs_
                ret = ret_

    clusterer1 = best_clusterer
    um = best_um

    # Might save it for later if needed?
    if args.save_mod:
       print("Saving UMAP and HDBSCAN models...")
       pickle.dump(um, open(os.path.join('..', 'models', args.dataset, 'umap_{}_{}_{}.npz'.format(args.type, args.obj_model, args.ind)), 'wb'))
       pickle.dump(best_clusterer, open(os.path.join('..', 'models', args.dataset, 'hdbscan_{}_{}_{}.npz'.format(args.type, args.obj_model, args.ind)), 'wb'))

    print("########### Summary Best Models ###########")
    print("Min dist: {}, Number of components: {}, Number of neighbors: {}".format(best['n_n'], best['i'], best['k']))
    print("Epsilon: {}, Min sample size: {}, Min cluster size: {}".format(best['epsilon'], best['m_samples'], best['m_size']))
    print("Nb of cluster generated: {} ({} covered by given test set of type {})".format(total_nb_cluster, nb_covered_clusters_obj, args.type))
    print("silhouette_score: {}, DBCV score: {}, Percentage noisy inputs in test set: {:.2f}".format(shs, dbcv_s, noisy_inputs/len(label_data)*100))
    print("\n")
    
    # Writing the paramters of best models
    with open('{}_cluster.txt'.format(args.type if args.dataset == 'cifar10' else "text"), 'a') as f:
        f.writelines("Obj: {}_{} ".format(args.obj_model, args.ind))
        f.writelines("Min dist: {}, Number of components: {}, Number of neighbors: {}".format(best['n_n'], best['i'], best['k']))
        f.writelines("Epsilon: {}, Min sample size: {}, Min cluster size: {}".format(best['epsilon'], best['m_samples'], best['m_size']))
        f.writelines("Nb of cluster generated: {} ({} covered by given test set of type {}) ".format(total_nb_cluster, nb_covered_clusters_obj, args.type))
        f.writelines("silhouette_score: {}, DBCV score: {}, Percentage noisy inputs in test set: {:.2f}".format(shs, dbcv_s, noisy_inputs/len(label_data)*100))
        f.writelines("\n")
    
    
    # Dict with all relevant information
    nb_cov_clust_dict = {'silhouette_score': float(shs), 
                        'dbcv': float(dbcv_s), 
                        'total_nb_cluster': total_nb_cluster, 
                        'nb_covered_clusters_obj': nb_covered_clusters_obj,
                        'noisy_inputs_pct': float(noisy_inputs/len(label_data)*100),
                        'cluster_all': ret.tolist(),
                        'mod_lab': mod_lab,
                        'Xmin': X_min.tolist(),
                        'Xmax': X_max.tolist(),
                        }

    mod_lab = np.array(mod_lab)

    for u in np.unique(mod_lab):
        if u == 'obj':
            nb_cov_clust_dict['cluster_per_seed'] = ret[np.where(mod_lab == u)[0]].tolist()
            nb_cov_clust_dict['list_covered_clusters_obj'] = np.unique(nb_cov_clust_dict['cluster_per_seed']).tolist()
            if -1 in nb_cov_clust_dict['list_covered_clusters_obj']:
                nb_cov_clust_dict['list_covered_clusters_obj'].remove(-1)
        else:
            temp = u.split('_')
            nb_cov_clust_dict['cluster_per_seed_'+temp[0]+'_'+str(temp[1])] = ret[np.where(mod_lab == u)[0]].tolist()
            nb_cov_clust_dict['list_covered_clusters_ref_'+temp[0]+'_'+str(temp[1])] = np.unique(nb_cov_clust_dict['cluster_per_seed_'+temp[0]+'_'+str(temp[1])]).tolist()
            if -1 in nb_cov_clust_dict['list_covered_clusters_ref_'+temp[0]+'_'+str(temp[1])]:
                nb_cov_clust_dict['list_covered_clusters_ref_'+temp[0]+'_'+str(temp[1])].remove(-1)
    
    # Saving the obtain clusters
    with open(os.path.join('..', 'pred_sets', args.dataset, 'cov_type2_cluster_{}_{}_{}.json'.format(args.obj_model, args.ind, args.type)), "w") as f:
                json.dump(nb_cov_clust_dict, f)
    
    # Saving the data per clusters
    # Just need for the Clustering Validation part 
    if args.save_clust:
       np.savez(os.path.join('..', 'data', args.dataset, 'data_cov_{}_{}_{}.npz'.format(args.obj_model, args.ind, args.type)), advs=global_orig_data, labels=ground_truth, pred=label_data)
