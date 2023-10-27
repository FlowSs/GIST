# Utils to convert the .csv results files to a proper numpy array
# Only look at the successful attacks
import pandas as pd
import numpy as np
import re
import os

onlyfiles = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f)) and 'csv' in f]

for f in onlyfiles:
   my_data = pd.read_csv(f)
   data_good = my_data[my_data['result_type'] == 'Successful']

   corrupted_text = [re.sub(r"(\[\[|\]\])", "", l) for l in data_good['perturbed_text']]
   labels = data_good['ground_truth_output']

   np.savez('{}.npz'.format(f[:-4]), text=corrupted_text, label=labels)

