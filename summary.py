#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 07:24:14 2022

@author: menghanlin

This file summarizes number of atoms in protein env near the ligand of interest
and their type. 
First generate data (pdbid, ligand_name, atom_type, atom_number)
"""

import pickle 
import numpy as np
import pandas as pd
import time
from sklearn.metrics.pairwise import pairwise_distances

ligand_len = pickle.load( open( "./ligand_len.pkl", "rb" ))
file = pickle.load( open( "./all_ligand_atom_type.pkl", "rb" ))

def read_pdb(file_name):
    '''
    Read protein env
    ! discard atom H;
    ! Skip alternative positions

    :param file_name: './data/xxxx.pdb'
    :return: [[label,x,y,z],...]
    '''
    data = []
    atom = []
    with open (file_name) as f:
        for line in f:
            if line[0:6]=='ATOM  ':
                atomName = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                altLoc = line[16]
                atomSym = line[76:78].strip()
                resName = line[17:20]
                if atomSym != 'H':
                    # The corresponding atom channel, given residual name and atom type
                    if altLoc == ' ' or altLoc == 'A':  # skip alternate position
                        data.append([x,y,z])
                        atom.append(atomName)
    return atom, data

# Keep valid ligands
# Keep env atoms close to ligand

THRESHOLD = 6
data = []
count_valid = 0
count_ligands = 0
ligand_name_cache = []
t1 = time.time()
for k, v in file.items():
    try:
        ligand_name = k
        ligand_length = ligand_len[k]
        N = len(v)
        for n in range(N):
            # ligand axes
            pdbid = v[n][0]
            axes = v[n][2]
            count_ligands += 1
            if len(v[n][1]) == ligand_length: # then no missing atoms
                count_valid += 1
                
                # protein axes
                file_name = './data/'+pdbid.lower()+'.pdb'
                env_atom, env_axes = read_pdb(file_name)
                dist = pairwise_distances(axes,env_axes)
                dist = np.sum(dist < THRESHOLD,0) > 0
                
                atom = v[n][1]
                env_atom = [env_atom[i] for i,v in enumerate(dist) if v == True]
                atom_type = set(env_atom)
                atom_number = len(env_atom)
                data.append((pdbid, ligand_name, atom_type, atom_number, ligand_length))
                
                if (count_valid % 1000)==0:
                    print(time.time()-t1)
                    print('Number of ligand left:',len(file) - count_ligands)
    except:
        print(ligand_name)
        ligand_name_cache.append(ligand_name)
                
count_valid/count_ligands # 0.811

temp = pd.DataFrame(data)
temp.columns = ['pdbid','ligand_name','atom_type','atom_num','ligand_len']
temp.to_csv('ligand_env.csv')


# Statistics ------------------------------------------------------------------
import pandas as pd
import numpy as np
temp = pd.read_csv('ligand_env.csv', index_col=[0])
temp.columns = ['pdbid','ligand_name','env_type','env_len','ligand_len']

# ligand length summary
temp1 = temp[['ligand_name','ligand_len']].drop_duplicates()
temp1.hist()


# Type of atoms in the env
base_set = set()

for i in range(len(temp)):
    result = temp['env_type'][i].replace("'","")
    result = result.replace("{","")
    result = result.replace("}","")
    result = result.replace(",","")
    result = result.replace('"','')
    result = result.split(' ')
    base_set = set.union(base_set, set(result))
base_set = set([i for i in base_set if i!='set()'])
        
# Exclude atoms starting with D
base_set = sorted([i for i in base_set if i[0] != 'D'])
print('All types of atoms in protein env:')
print(base_set)      
print('Summary stats of the number of atoms in protein env (ligand length = 40-50):')
temp[(temp['ligand_len']>=40)&(temp['ligand_len']<=50)]['env_len'].describe()
         
        
        
        
        
        
        
        
        
        
        
        



