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


s_possible = np.asarray([ 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 24, 25, 26,
                27, 30, 31, 34, 35, 38, 39, 44, 45, 48, 49, 50, 54, 55, 58, 59, 62,
                64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84, 85, 88, 89, 94, 95, 98,
                99])


s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1

madx,posz = place_quads_wmarkers(s_operational,s_possible,ssz)

std_min = get_mean_optics(madx)
try:
    while 1:
        
        pos_possible = np.asarray(posz)[s_possible]
        pos_operational = np.asarray(posz)[s_operational]
        
        A=[madx.eval('kf'),madx.eval('kd')]/(2*np.sin(2*np.pi*6.1))
        Bx_err = [-A[s_possible[i]%2]*madx.table.twiss.betx*madx.table.twiss.betx[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.mux[pos] - madx.table.twiss.mux) -2*np.pi*6.1) for i,pos in enumerate(pos_possible)]                
        By_err = [A[s_possible[i]%2]*madx.table.twiss.bety*madx.table.twiss.bety[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.muy[pos] - madx.table.twiss.muy) -2*np.pi*6.1) for i,pos in enumerate(pos_possible)]
        Dx_err = [-(np.sin(2*np.pi*6.1)/ np.sin(np.pi*6.1))*A[s_possible[i]%2]*madx.table.twiss.dx*madx.table.twiss.betx[pos]*np.cos(2*np.pi*np.abs(madx.table.twiss.mux[pos] - madx.table.twiss.mux) -2*np.pi*6.1) for i,pos in enumerate(pos_possible)]
         
        temp = [i for i, elem in enumerate(s_possible) if elem in s_operational]
        
        Bx_err.extend([-Bx_err[s] for s in temp])
        By_err.extend([-By_err[s] for s in temp])
        Dx_err.extend([-Dx_err[s] for s in temp])
        
        Bx_diff = [madx.table.twiss.betx + err for err in Bx_err]
        By_diff = [madx.table.twiss.bety + err for err in By_err]
        Dx_diff = [madx.table.twiss.dx + err for err in Dx_err]    
        
        total_std = np.asarray([get_mean_optics2(madx.table.twiss.s.T,Bx_diff[i],By_diff[i],Dx_diff[i]) for i in range(len(Bx_err))])
        
        if len(s_operational) == 40:
            total_std = total_std[len(s_possible):]        #check this
            idx = total_std.argmin()
            idx = idx + len(s_possible)
        else:
            idx = total_std.argmin()

        
        if idx >=len(s_possible):
            madx,flag = remove_wmarkers(madx,s_operational,s_operational[idx-len(s_possible)])
            temp3 = s_operational[idx-len(s_possible)]
            s_operational = np.delete(s_operational, idx-len(s_possible))
            
        else:
            madx,flag = add_wmarkers(madx,s_operational,s_possible[idx])
            temp3 = s_possible[idx]
            s_operational = np.append(s_operational,s_possible[idx])
            
          
        
        temp2 = counter(s_operational)
        print(temp2)
        
        
        actual_std = get_mean_optics(madx)
        if actual_std > std_min:
            print('old std:', std_min)
            print('new std found:', actual_std)
            print('#---------------------------------')
            print(s_operational, idx)
            break
        else:
            print('sector added or removed:', temp3)
            print('old std:', std_min)
            print('new std found:', actual_std)
            print('#---------------------------------')
            std_min = actual_std
        
        
except ValueError:
    print('error')

    

