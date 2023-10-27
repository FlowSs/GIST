# Standard Library
import os
import argparse
import re

# Public Library
import torch
import torchvision
import transformers
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

# Custom
from architecture import *
from const import REGISTERED_COMBINATIONS

class CustomDataset(torch.utils.data.Dataset):
   def __init__(self, data, lab, tokenizer=None, max_len=None):
      self.data = data
      self.lab = lab

      # Tokenizer is not None only for text data
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

def prep_dataset(dataset, batch_size, tp, models_choice, target_model):

   if dataset == 'mr':
         testloader = {}
         
         for mod in models_choice:
          #print(mod)
          testloader[mod] = []
          tokenizer_mod = mod if target_model is None else target_model.split('_')[0]
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
         
          files = os.listdir(os.path.join('..', 'models', args.dataset))
          file_list = [f for f in files if re.match(re.escape(mod)+'_[0-9]{1}'+re.escape('.'), f) is not None]

          for my_f in file_list:
             # If target provided, skip it
             i = int(my_f.split('_')[1].split('.')[0])
             if mod + '_' + str(i) == target_model:
                testloader[mod].append([])
                continue
             
             if tp == 'train':
               target_data = np.load(os.path.join('..', 'data', args.dataset, f'{tp}.npz'))
             else:
               target_data = np.load(os.path.join('..', 'data', args.dataset, 'gen', 'log_' + mod + '_' + str(i) + '.npz'))
               print('Gen dataset size for model {} is {} '.format(mod+'_'+str(i), len(target_data['label'])))
             
             custom_dataset = CustomDataset(target_data['text'], target_data['label'], tokenizer=tokenizer, max_len=128)             
             testloader[mod].append(torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0))
   else:
      if tp == 'train':
        if dataset == 'cifar10':
         # For now, one transform
         transform = torchvision.transforms.Compose(
           [torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
         trainset = torchvision.datasets.CIFAR10(root='../data/{}/'.format(dataset), train=True,
                                          download=True, transform=transform)
         testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=False, num_workers=8)
         
        else:
           raise Exception("Dataset {} not recognised".format(dataset))

      elif tp == 'fuzz':
       testloader = {}         

       for mod in models_choice:
        testloader[mod] = []
        files = os.listdir(os.path.join('..', 'models', args.dataset))
        file_list = [f for f in files if re.match(re.escape(mod)+'_[0-9]{1}'+re.escape('.'), f) is not None]
        
        for my_f in file_list:
           # If target provided, skip it
           i = int(my_f.split('_')[1].split('.')[0])
           if mod + '_' + str(i) == target_model:
               testloader[mod].append([])
               continue
           target_data = np.load(os.path.join('..', 'data', args.dataset, 'fuzz_sets', mod + '_' + str(i) + '_fuzz_results.npz'))
           custom_dataset = CustomDataset(target_data['advs'], target_data['labels'])
           print('Fuzz dataset size for model {} is {} '.format(mod+'_'+str(i), len(custom_dataset)))
           testloader[mod].append(torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=8))
      elif tp =='gen':
       testloader = {}   
       
       for mod in models_choice:
          testloader[mod] = []
          files = os.listdir(os.path.join('..', 'models', args.dataset))
          file_list = [f for f in files if re.match(re.escape(mod)+'_[0-9]{1}'+re.escape('.'), f) is not None]

          for my_f in file_list:
             # If target provided, skip it
             i = int(my_f.split('_')[1].split('.')[0])
             if mod + '_' + str(i) == target_model:
                testloader[mod].append([])
                continue
             target_data = np.load(os.path.join('..', 'data', args.dataset, 'gan_sets', mod + '_' + str(i) + '_gen_results.npz'))
             # Manual labels since target_data['labels'] give the label PREDICTED, when we need the 'original' label needed
             # Hardcoded since we have 1,000 samples
             custom_dataset = CustomDataset(target_data['advs'], np.concatenate([[i]*100 for i in range(10)]))
             print('GAN dataset size for model {} is {} '.format(mod+'_'+str(i), len(custom_dataset)))
             testloader[mod].append(torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=8))

   return testloader

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-m', '--model', help='Model to use for computation. By default, will use ALL registered model for the given dataset. If given, \
      will use only the provided list of models (e.g. --model preresnet20 vgg19). All instances (i.e. seed) trained are used, e.g. --model vgg19 will use vgg19_0, vgg19_1, ...', nargs='+', default=None)
   parser.add_argument('-d', '--dataset', help='Dataset to use for computation', default='cifar10')
   parser.add_argument('-t', '--type', help='Test set type (train, fuzz, gen)', default=None)
   parser.add_argument('-tm', '--target_model', help='Model to be tested on generated dataset of all models provided with --model argument, e.g --target_model vgg19_0, will test generated dataset from all instances of \
      all models seeds (except vgg19 ones) on vgg19_0.', default=None)
   parser.add_argument('-b', '--batch_size', type=int, default=128)
   parser.add_argument('--device', default='cpu')
   parser.add_argument('--override', action='store_true')
   args = parser.parse_args()

   assert args.dataset in REGISTERED_COMBINATIONS, "Dataset {} not recognised".format(args.dataset)
   
   if args.override:
      print("Warning: override argument has been provided, will override already calculated pred sets...")
   
   if args.model is not None:
      models_choice = []
      for mod in args.model:
         assert mod in REGISTERED_COMBINATIONS[args.dataset], "Model {} not recognised for dataset {}".format(mod, args.dataset)
         models_choice.append(mod)
   else:
      models_choice = REGISTERED_COMBINATIONS[args.dataset]
      if args.target_model is not None:
         models_choice.remove(args.target_model.split('_')[0])

   if torch.cuda.is_available and 'cuda' not in args.device:
      print("Warning: CUDA is available but not used currently")

   testloader = prep_dataset(args.dataset, args.batch_size, args.type, models_choice, args.target_model)
   for mod in models_choice:
      
      files = os.listdir(os.path.join('..', 'models', args.dataset))
      file_list = [f for f in files if re.match(re.escape(mod)+'_[0-9]{1}'+re.escape('.'), f) is not None] # and args.target_model is None]) # (args.target_model is None or args.target_model != f)])
      for my_f in tqdm(file_list):
         i = int(my_f.split('_')[1].split('.')[0])
         # Should not compute on the model itself. 
         # By design, should not happen but just in case!
         if mod + '_' + str(i) == args.target_model:
            continue
         print("Running on {}...".format(mod))
         if os.path.exists(os.path.join('..', 'pred_sets', args.dataset, '{}_{}.npz'.format(mod + '_' + str(i), \
                           args.type if args.target_model is None else args.type + '_' + args.target_model))) and not(args.override):
               print("Already computed...")
               if args.target_model is not None:
                  break
               continue
         
         dict_glob = None
         if args.target_model is None:
            state_dict_ = torch.load(os.path.join('..', 'models', args.dataset, '{}_{}{}'.format(mod, i, '.pth.tar_best.pth.tar' \
              if args.dataset == 'cifar10' else '.best.pth')))
            state_dict_ = state_dict_['state_dict']
         else:
            state_dict_ = torch.load(os.path.join('..', 'models', args.dataset, '{}{}'.format(args.target_model, '.pth.tar_best.pth.tar' \
              if args.dataset == 'cifar10' else '.best.pth')))
            state_dict_ = state_dict_['state_dict']

         # Refactoring the layers name so it can be loaded properly (by default, the CIFAR-ZOO library seems to add "module." for all the layer name?)
         # TODO: preprocess it on my own or move the function somewhere
         if args.dataset == 'cifar10':
            for k in list(state_dict_.keys()):
               new_k = '.'.join(k.split('.')[1:])
               state_dict_[new_k] = state_dict_[k]
               del state_dict_[k]

         if args.dataset != 'mr':
            # Loading classifier to generate pred set on
            if args.target_model is None:
               my_classifier = eval(mod)(num_classes=10, input_channel=3)
            else:
               my_classifier = eval(args.target_model.split('_')[0])(num_classes=10, input_channel=3)
         else:
            if args.target_model is None:
               # get max len
               my_classifier = eval(mod)()
            else:
               my_classifier = eval(args.target_model.split('_')[0])()

         my_classifier.load_state_dict(state_dict_)
         my_classifier.to(args.device)
         my_classifier.eval()
         
         if args.type == 'train' and args.dataset != 'mr':
            curr_testloader = testloader
         else:
            curr_testloader = testloader[mod][i]
         
         for (img, lab) in curr_testloader:
            if args.dataset == 'mr':
               input_dict = {'input_ids': torch.stack(img['input_ids']).permute(1, 0).to(args.device), \
               'attention_mask': torch.stack(img['attention_mask']).permute(1, 0).to(args.device), 'hook': True}
            else:
               input_dict = {'x': img.to(args.device), 'hook': True}
            
            with torch.no_grad():
             if args.type == 'train':
              dict_out = my_classifier(**input_dict)
              miss = np.array([1 if res else -1 for res in np.equal(np.argmax(dict_out['pred'], 1), 
                                                lab)], dtype=np.float32)
              dict_out['pred_before'] = dict_out['pred']
              miss = miss[:, None] * softmax(dict_out['pred'],1)
              dict_out['pred'] = miss
              if dict_glob is None:
                dict_glob = dict_out
              else:
                dict_glob = {key: np.concatenate((value, dict_out[key]), axis=0) for key,value in dict_glob.items()}
             else:
              dict_out = my_classifier(**input_dict)
              miss = np.array([1 if res else -1 for res in np.equal(np.argmax(dict_out['pred'], 1), 
                                                lab)], dtype=np.float32)
              miss = miss[:, None] * softmax(dict_out['pred'],1)
              dict_out['pred'] = miss
              if dict_glob is None:
                dict_glob = dict_out
              else:
                dict_glob = {key: np.concatenate((value, dict_out[key]), axis=0) for key,value in dict_glob.items()}
         
         assert len(dict_glob['pred']) == len(curr_testloader.dataset), "{} vs {}".format(len(dict_glob['pred']), len(curr_testloader.dataset))
         if not os.path.exists(os.path.join('..', 'pred_sets', args.dataset)):
            os.mkdir(os.path.join('..', 'pred_sets', args.dataset))
         
         np.savez(os.path.join('..', 'pred_sets', args.dataset, '{}_{}'.format(mod+'_'+ str(i), \
                             args.type if args.target_model is None else args.type + '_' + args.target_model)), **dict_glob)






