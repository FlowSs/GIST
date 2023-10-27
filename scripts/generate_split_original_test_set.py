# Standard library
import os
import json
import argparse

# Public library
import numpy as np

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
   parser.add_argument('-d', '--dataset', help='Dataset to use for computation', default='cifar10')
   parser.add_argument('-t', '--type', help='Test set type (train, test, fuzz, gen)', default='test')
   parser.add_argument('-o', '--obj_model', help='Objective model to compare to reference', default=None)
   parser.add_argument('-i', '--ind', help='Random instance among objective model to use', default=0, type=int)
   parser.add_argument('-c', '--cluster', help='Cluster number (sorted by number of faults) to use', default=None, type=int)
   args = parser.parse_args()

   with open(os.path.join('..', 'pred_sets', args.dataset, 'cov_type2_cluster_{}_{}_{}.json'.format(args.obj_model, args.ind, args.type)), "r") as f:
         cluster_info = json.load(f)      

   l_ , faults_per_cluster = np.unique(np.array(cluster_info['cluster_all']), return_counts=True)
   cluster_per_seed = np.array(cluster_info['cluster_all'])
   l_ = l_.tolist()
   # Removing noise
   if -1 in l_:
      l_.remove(-1)
      faults_per_cluster = faults_per_cluster[1:]

   ordered_cluster = np.array(l_)[np.argsort(faults_per_cluster)][::-1]
   clust = args.cluster
   print(ordered_cluster[clust])
   l, orig_mod = np.unique(np.array(cluster_info['cluster_per_seed']), return_counts=True)
   print(orig_mod[(np.where(l == ordered_cluster[clust])[0])])

   my_dat = np.load(os.path.join('..', 'data', args.dataset, 'data_cov_{}_{}_{}.npz'.format(args.obj_model, args.ind, args.type)))

   # Seeding to have the same split
   rng = np.random.default_rng(42)

   for i in range(3):
      tmp = np.where(np.array(cluster_info['cluster_all']) == ordered_cluster[clust])[0]
      seeds = rng.choice(tmp, size=int(0.85*len(tmp)), replace=False)
      print(len(seeds), len(tmp))
      non_seeds = list(set(np.arange(len(cluster_info['cluster_all']))) - set(seeds))

      dat_adv_list = my_dat['advs'][seeds]
      dat_label_list = my_dat['labels'][seeds]

      print(dat_adv_list.shape, dat_label_list.shape)
      print(my_dat['advs'].shape, my_dat['labels'].shape)

      # TODO: might need to unify name not to have both 'gan' and 'gen' in the file name
      np.savez(os.path.join('..', 'data', args.dataset, '{}_sets'.format('fuzz' if args.type == 'fuzz' else 'gan') if args.dataset == 'cifar10' else 'gen',\
         '{}_{}_{}_clust_{}_split_{}_results.npz'.format(args.obj_model, args.ind, args.type, ordered_cluster[clust], i)),\
            advs=dat_adv_list, labels=dat_label_list)
      np.savez(os.path.join('..', 'data', args.dataset, 'clust_{}_split_{}_{}_{}_{}.npz'.format(ordered_cluster[clust], i, args.type, args.obj_model, args.ind)), seeds = seeds, non_seeds=non_seeds)