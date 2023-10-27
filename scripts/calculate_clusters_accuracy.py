# Standard library
import os
import argparse
import json
import random

# Public library
import torch
import transformers
import numpy as np
from tqdm import tqdm

# Custom
from architecture import *
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

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class CustomDataset(torch.utils.data.Dataset):
   def __init__(self, data, lab, tokenizer=None, max_len=None):
      self.data = data
      self.lab = lab

      if tokenizer is not None:
         self.transform = lambda x: (tokenizer(x[0], padding='max_length', max_length=max_len, truncation='longest_first'), x[1])
      else:
         self.transform = None

   # Getting the data samples
   def __getitem__(self, idx):
      sample = self.data[idx], self.lab[idx]

      if self.transform:
         sample = self.transform(sample)
      return sample
    
   def __len__(self):
      return len(self.lab)

def prep_dataset(dataset, batch_size, tp, target_model, ind):

   if dataset == 'mr':
         testloader = {}
         
         testloader = []
         tokenizer_mod = target_model.split('_')[0]
         if tokenizer_mod == 'bert':
            tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
         elif tokenizer_mod == 'distill':
            tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
         elif tokenizer_mod == 'electra':
            tokenizer = transformers.AutoTokenizer.from_pretrained("google/electra-small-discriminator", do_lower_case=True)
         elif tokenizer_mod == 'xlnet':
            tokenizer = transformers.AutoTokenizer.from_pretrained("xlnet-base-cased", do_lower_case=True)
         elif tokenizer_mod == 'roberta':
            tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base", do_lower_case=True)
         else:
            raise Exception("Model {} not in the list of registered model {}".format(target_model, REGISTERED_COMBINATIONS['mr']))
         
         target_data = np.load(os.path.join('..', 'data', args.dataset, 'data_cov_{}_{}_{}.npz'.format(target_model, ind, tp)))
         print('Gen dataset size for model {} is {} '.format(target_model+'_'+str(ind), len(target_data['labels'])))
             
         custom_dataset = CustomDataset(target_data['advs'], target_data['labels'], tokenizer=tokenizer, max_len=128)             
         testloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
         return testloader, target_data['labels']
   else:
      if tp == 'fuzz':       
       target_data = np.load(os.path.join('..', 'data', args.dataset, 'data_cov_{}_{}_{}.npz'.format(target_model, ind, tp)))
       custom_dataset = CustomDataset(target_data['advs'], target_data['labels'])
       target_labels = target_data['labels']
       print('Fuzz dataset size for model {} is {} '.format(target_model+'_'+str(ind), len(custom_dataset)))
       testloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)
      elif tp =='gen':
       target_data = np.load(os.path.join('..', 'data', args.dataset, 'data_cov_{}_{}_{}.npz'.format(target_model, ind, tp)))
       custom_dataset = CustomDataset(target_data['advs'], target_data['labels'])
       target_labels = target_data['labels']
       print('GAN dataset size for model {} is {} '.format(target_model+'_'+str(ind), len(custom_dataset)))
       testloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

      return testloader, target_labels

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-d', '--dataset', help='Dataset to use for computation', default='cifar10')
   parser.add_argument('-t', '--type', help='Test set type (train, test, fuzz, gen)', default='test')
   parser.add_argument('-o', '--obj_model', help='Objective model to compare to reference', default=None)
   parser.add_argument('-i', '--ind', help='Random instance among objective model to use', default=0, type=int)
   parser.add_argument('-b', '--batch_size', type=int, default=128)
   parser.add_argument('--cluster', default=None)
   parser.add_argument('--device', default='cpu')
   args = parser.parse_args()

   assert args.dataset in REGISTERED_COMBINATIONS, "Dataset {} not recognised".format(args.dataset)
   assert args.obj_model is not None, "An objective model must be specified!"

   if torch.cuda.is_available and 'cuda' not in args.device:
      print("Warning: CUDA is available but not used currently")
   
   print("Calculating...")
   with open(os.path.join('..', 'pred_sets', args.dataset, 'cov_type2_cluster_{}_{}_{}.json'.format(args.obj_model, args.ind, args.type)), "r") as f:
     cluster_info = json.load(f)
   
   pct_covered_faults_in_list, pct_covered_faults_out_list = [], []

   for split in range(3):
      print("Working on split {}".format(split))
      my_split = np.load(os.path.join('..', 'data', args.dataset, 'clust_{}_split_{}_{}_{}_{}.npz'.format(args.cluster, split, args.type, args.obj_model, args.ind)))['non_seeds']

      l_ , faults_per_cluster = np.unique(np.array(cluster_info['cluster_all'])[my_split], return_counts=True)
      l_ = l_.tolist()
      if -1 in l_:
         l_.remove(-1)
         faults_per_cluster = faults_per_cluster[1:]

      testloader, ground_truth = prep_dataset(args.dataset, args.batch_size, args.type, args.obj_model, args.ind)
         
      avg_proportion = 0
      files = os.listdir(os.path.join('..', 'models', args.dataset))
      list_f = [f for f in files if args.type in f and ('clust_' + str(args.cluster) +'_split_' + str(split) in f) and f.startswith(args.obj_model + '_' + str(args.ind)) and not(f.startswith(args.obj_model + '_' + str(args.ind) + '.'))]
      
      state_dict_ = torch.load(os.path.join('..', 'models', args.dataset, list_f[0]))
      state_dict_ = state_dict_['state_dict']

      # Refactoring the layers name so it can be loaded properly (by default, the CIFAR-ZOO library seems to add "module." for all the layer name?)
      if args.dataset == 'cifar10':
         for k in list(state_dict_.keys()):
            if k.split('.')[0] == 'module':
               new_k = '.'.join(k.split('.')[1:])
               state_dict_[new_k] = state_dict_[k]
               del state_dict_[k]

      if args.dataset != 'mr':
         # Loading classifier to generate pred set on
         my_classifier = eval(args.obj_model)(num_classes=10, input_channel=3)
      else:
         my_classifier = eval(args.obj_model)()

      my_classifier.load_state_dict(state_dict_)
      my_classifier.to(args.device)
      my_classifier.eval()
               
      curr_testloader = testloader
      temp_dict = []
      for (img, lab) in curr_testloader:
         if args.dataset == 'mr':
            input_dict = {'input_ids': torch.stack(img['input_ids']).permute(1, 0).to(args.device), \
                     'attention_mask': torch.stack(img['attention_mask']).permute(1, 0).to(args.device), 'hook': True}
         else:
            input_dict = {'x': img.to(args.device), 'hook': True}
                  
         with torch.no_grad():
            dict_out = my_classifier(**input_dict)
            miss = [1 if res else -1 for res in np.equal(np.argmax(dict_out['pred'], 1), 
                                                      lab)]
            del dict_out
            temp_dict.extend(miss)

               
      misc_inp_norm = np.intersect1d(np.where(np.array(temp_dict) == -1)[0], my_split)
      l, non_covered_faults = np.unique(np.array(cluster_info['cluster_all'])[misc_inp_norm], return_counts=True)
      l = l.tolist()
               
      # Removing noise cluster
      if -1 in l:
         l.remove(-1)
         non_covered_faults = non_covered_faults[1:]
               
      pct_covered_faults = []
      pct_covered_faults_in, pct_covered_faults_out = [], []
      # For each fault types in the objective model test set
      for f in l_:
         # if the fault type is still being done after retraining
         if f in l:
            pct_covered_faults.append(float(non_covered_faults[l.index(f)]/faults_per_cluster[l_.index(f)]))
            # Is this the fault type of the cluster we used a part of for retraining?
            if f == int(args.cluster):
               pct_covered_faults_in.append(float(non_covered_faults[l.index(f)]/faults_per_cluster[l_.index(f)]))
            else:
               pct_covered_faults_out.append(float(non_covered_faults[l.index(f)]/faults_per_cluster[l_.index(f)]))

         else:
            pct_covered_faults.append(0)
            if f == int(args.cluster):
               pct_covered_faults_in.append(0)
            else:
               pct_covered_faults_out.append(0)
                  
      pct_covered_faults_in_list.append(1 - np.mean(np.array(pct_covered_faults_in)))
      pct_covered_faults_out_list.append(1 - np.mean(np.array(pct_covered_faults_out)))

   print("Using cluster {}, average over 3 splits".format(args.cluster))
   print("Averaged pct of faults reduced (non-covered cluster) in dat aug: ", np.mean(np.array(pct_covered_faults_in_list)))
   print("Averaged pct of faults reduced (non-covered cluster) not in dat aug: ", np.mean(np.array(pct_covered_faults_out_list)))
   print("#####################################")
         










