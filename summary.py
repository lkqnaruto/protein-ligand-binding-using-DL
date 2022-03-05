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
         
        
        
        
## New function----------------------------------------------------------------
# ligand_name, pdbid,(aligned_dataset) 
# ligand_axes [22, x, y, z], atom_axes[atom_type[dict], ax, ay, az]
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
    resNames = []
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
                        resNames.append(resName)
    return atom, data, resNames

def label_atom(resname,atom):
    atom_label={
    'ARG':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':4,'NE':11,'CZ':1,'NH1':11,'NH2':11},
    'HIS':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD2':6,'CE1':6,'ND1':8,'NE2':8},
    'LYS':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':4,'CE':4,'NZ':10},
    'ASP':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'OD1':15,'OD2':15},
    'GLU':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':2,'OE1':15,'OE2':15},
    'SER':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'OG':13},
    'THR':{'N':17,'CA':18,'C':19,'O':20,'CB':3,'OG1':13,'CG2':5},
    'ASN':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':1,'OD1':14,'ND2':9},
    'GLN':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':1,'OE1':14,'NE2':9},
    'CYS':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'SG':16},
    'GLY':{'N':17,'CA':18,'C':19,'O':20},
    'PRO':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':4},
    'ALA':{'N':17,'CA':18,'C':19,'O':20,'CB':5},
    'VAL':{'N':17,'CA':18,'C':19,'O':20,'CB':3,'CG1':5,'CG2':5},
    'ILE':{'N':17,'CA':18,'C':19,'O':20,'CB':3,'CG1':4,'CG2':5,'CD1':9},
    'LEU':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':3,'CD1':5,'CD2':5},
    'MET':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'SD':16,'CE':5},
    'PHE':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD1':6,'CD2':6,'CE1':6,'CE2':6,'CZ':6},
    'TYR':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD1':6,'CD2':6,'CE1':6,'CE2':6,'CZ':6,'OH':13},
    'TRP':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD1':6,'CD2':6,'NE1':7,'CE2':6,'CE3':6,'CZ2':6,'CZ3':6,'CH2':6}
    }
    if atom in ['OP1', 'OP2', 'OXT', 'P']:
        return 21   # define the extra oxygen atom OXT on the terminal carboxyl group as 21 instead of 27 (changed on March 19, 2018)
    
    else:
        if atom in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']:
            atom = 'C'
        if atom in ['N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9']:
            atom = 'N'
        if atom in ['O2', 'O3', 'O4', 'O5', 'O6']:
            atom  = 'O'
        return atom_label[resname][atom]


THRESHOLD = 6
data = []
count_ligands = 0
ligand_name_cache = []
t1 = time.time()
for k, v in file.items():
    try:
        ligand_name = k
        ligand_length = ligand_len[k]
        count_ligands += 1
        N = len(v)
        for n in range(N):
            # ligand axes
            pdbid = v[n][0]
            ligand_coords = v[n][2] # coords of ligand atoms
            if len(v[n][1]) == ligand_length: # then no missing atoms
                
                # protein coords
                file_name = './data/'+pdbid.lower()+'.pdb'
                env_atom, env_coords, resName = read_pdb(file_name)
                dist = pairwise_distances(ligand_coords, env_coords)
                dist = np.sum(dist < THRESHOLD,0) > 0
                
                # ligand_axes [22, x, y, z], atom_axes[atom_type[dict], ax, ay, az]
                env_coords = [[label_atom(resName[i], env_atom[i]),*env_coords[i]] for i,v in enumerate(dist) if v == True]
                ligand_coords = [[22, *i] for i in ligand_coords]
                data.append((pdbid, ligand_name, ligand_coords, env_coords))
                
        if (count_ligands % 1000)==0:
            print(time.time()-t1)
            print('Number of ligand left:',len(file) - count_ligands)
    except:
        print(ligand_name)
        ligand_name_cache.append(ligand_name)
                
a_file = open("ligand_env_coords.pkl", "wb")
pickle.dump(data, a_file)
a_file.close()

temp = open("ligand_name_cache.pkl", "wb")
pickle.dump(ligand_name_cache, temp)
temp.close()        
        
        
        
        
        
        



