## 2. Keep VALID ligands (no missing atoms aka the LONGEST)
# Dictionary to decide valid length
# list(all_ligand_atom_type.items())[0]
# ligand_name = [k for k, v in all_ligand_length.items()]
# for k,v in all_ligand_length.items():
#     v = set([(i,j) for i,j in v])
#     v1 = [pdbid for (pdbid,length) in v]
#     v2 = [length for (pdbid, length) in v]
#     length = np.max(v2)
#     valid_pdbid = [i for (i,j) in zip(v1,idx) if j == True]

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:07:31 2022

@author: menghanlin
Extract ligand information from PDB, count atom length and  # H
Return a dictionary key = ligand name, value = # atoms - # H

"""
import requests
from bs4 import BeautifulSoup
import re
import pickle
import os
import time
import numpy as np

print('Here',os.getcwd())
file = pickle.load(open("./all_ligand_atom_type.pkl", "rb"))

def f_exact_ligand_len(ligand_name):
    URL = 'https://www.rcsb.org/ligand/' + ligand_name
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")

    # true length
    tag = soup.find_all('table')[1]
    tag = tag.find_all('tr')[2]
    length = int(tag.find('td').text)

    # H length
    tag = soup.find_all('table')[0]
    tag = tag.find_all('tr')[3]
    atom_type = tag.find('td').getText()
    m = re.search('H[\d]+', atom_type)
    
    tag = soup.find_all('table')[0]
    tag = tag.find_all('tr')[4]
    atom_type = tag.find('td').getText()
    m1 = re.search('H[\d]+', atom_type)
    
    if m is not None:
        return length - int(m.group(0)[1:])
    elif m1 is not None:
        return length - int(m1.group(0)[1:])
    elif (m is None) and (m1 is None):
        return length



# ligand_len = dict()
# counter = 0
# t1 = time.time()
# for i in set(file.keys()):
#     try:
#         ligand_len[i] = f_exact_ligand_len(i)
#         counter += 1
#         if (counter % 1000)==0:
#             print(time.time() - t1)
#     except:
#         print(i)



# Make up ligands with more molecules
ligand_len = pickle.load( open( "./ligand_len.pkl", "rb" ))
file = pickle.load( open( "./all_ligand_atom_type.pkl", "rb" ))


for i in set(file.keys()):
    t1 = time.time()
    molecules = i.split(' ')
    if (len(molecules)>1) & (ligand_len.get(i, None) is None):
        for j in molecules:
            ligand_len[j] = f_exact_ligand_len(j)
        temp  = [ligand_len[j] for j in molecules]
        ligand_len[i] = np.sum(temp)
        print(i)
        print(len(file) - len(ligand_len))
        print(time.time()-t1)
        
        
a_file = open("ligand_len.pkl", "wb")
pickle.dump(ligand_len, a_file)
a_file.close()














