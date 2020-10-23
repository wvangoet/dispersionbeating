import numpy as np
import pickle
#import pandas
from zoopt import ExpOpt
from scipy.interpolate import interp1d
from cpymad.madx import Madx
from cl2pd import madx as cl2madx
import collections
from BB_lib import *

f1 = cl2madx.tfs2pd('pfw_25.txt')
Table1=f1.iloc[0].TABLE
table1 = Table1.to_numpy()

idx_ssz = []
ssz = []
delta = [1.486343, 2.886343]
for i,elem in enumerate(table1[1:,0]):
    if elem.startswith('PS') and elem.endswith('START'):
        if (elem[3] == '1') or (elem[3] == '6'):
            ssz.append(table1[i,2] + delta[1])
            idx_ssz.append(i)
        else:
            ssz.append(table1[i,2] + delta[0])
            idx_ssz.append(i)
ssz = np.asarray(np.sort(ssz))

phase_advances_x = np.asarray(Table1['MUX'][idx_ssz])*2*np.pi
phase_advances_y = np.asarray(Table1['MUY'][idx_ssz])*2*np.pi

s_possible = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
taken_pos = []
disp = []
start_pos = []
j = 1
start = 4

try:
    while len(taken_pos) < 40:
        start_pos.append(start)
        pos4,disp_effect = find_four_adept(start,taken_pos,s_possible,phase_advances_x,phase_advances_y,0.1)
        disp.append(disp_effect)
        if disp_effect == 1:
            start = np.sort([k for k in s_possible if not(k in taken_pos)])[j]
            j=j+1
        else:
            j = 1
            taken_pos.extend(pos4)
            start = min([k for k in s_possible if not(k in taken_pos)])
except:
    pass
print(disp)
print(start_pos)
print(taken_pos, len(taken_pos))
    
    
try_again = [k for k in s_possible if not(k in taken_pos)]
test_dict = {}
for k in try_again:
    pos4,disp_effect = find_four_adept(k,[],s_possible,phase_advances_x,phase_advances_y,0.175)
    test_dict[k] = [pos4,disp_effect]
    

