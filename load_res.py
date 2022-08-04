import numpy as np
import pdb
import codecs

file_name = "lr_seed.txt"
search_terms = ["theta lr", "phi lr"]

results =  {k:[] for k in search_terms}
pdb.set_trace()



with codecs.open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if "AVG. ELBO" and search_term in line:
            lsplit = line.split("\t")
            for unit in lsplit:
                key = unit.split(":")[0]
                val = unit.split(":")[1]
                if 'ELBO' in key or 'MCC' in key or i
                pdb.set_trace()

