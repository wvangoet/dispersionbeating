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

s_possible_2 = np.asarray([1,10,15,17,18,21,25,26,27,31,32,35,50,59,67,75,90])-1


#s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
s_operational = np.asarray([38, 34, 26, 30, 88, 84, 76, 80, 13, 17, 5, 9, 16, 12, 4, 8, 70, 66, 58, 62, 48, 44, 20, 24, 77, 81, 85, 89, 27, 31, 35, 39, 25, 21, 45, 49, 55, 59, 67, 95])


madx,posz = place_quads_wmarkers(s_operational,s_possible,ssz)

std_min = get_mean_optics(madx)
x=0
try:
    while x <=20:
        
        pos_possible = np.asarray(posz)[s_possible]
        pos_operational = np.asarray(posz)[s_operational]
        
        A=[madx.eval('kf'),madx.eval('kd')]/(2*np.sin(2*np.pi*6.1))
        Bx_err = [-A[s_possible[i]%2]*madx.table.twiss.betx*madx.table.twiss.betx[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.mux[pos] - madx.table.twiss.mux) -2*np.pi*6.1) for i,pos in enumerate(pos_possible)]                
        By_err = [A[s_possible[i]%2]*madx.table.twiss.bety*madx.table.twiss.bety[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.muy[pos] - madx.table.twiss.muy) -2*np.pi*6.1) for i,pos in enumerate(pos_possible)]
        Dx_err = [(np.sin(2*np.pi*6.1)/ np.sin(np.pi*6.1))*A[s_possible[i]%2]*madx.table.twiss.dx*madx.table.twiss.betx[pos]*np.cos(2*np.pi*np.abs(madx.table.twiss.mux[pos] - madx.table.twiss.mux) -2*np.pi*6.1) for i,pos in enumerate(pos_possible)]
         
        temp = [i for i, elem in enumerate(s_possible) if elem in s_operational]
        
        print('calculate possibilities')
        
        #[1=toevoegen, 2=verwijderen, 3=[1,2,..], 4 = [1,1,...], 5=[2,2,...]]
        
        Bx1 = np.asarray(Bx_err)
        Bx2 =  -Bx1[temp,:]
        Bx3 = np.asarray([Bx2 + Bx1[i,:] for i in range(len(Bx1))])
        Bx4 = np.asarray([Bx1 + Bx1[i,:] for i in range(len(Bx1))])
        Bx5 = np.asarray([Bx2 + Bx2[i,:] for i in range(len(Bx2))])
        
        By1 = np.asarray(By_err)
        By2 =  -By1[temp,:]
        By3 = np.asarray([By2 + By1[i,:] for i in range(len(By1))])
        By4 = np.asarray([By1 + By1[i,:] for i in range(len(By1))])
        By5 = np.asarray([By2 + By2[i,:] for i in range(len(By2))])
        
        Dx1 = np.asarray(Dx_err)
        Dx2 =  -Dx1[temp,:]
        Dx3 = np.asarray([Dx2 + Dx1[i,:] for i in range(len(Dx1))])
        Dx4 = np.asarray([Dx1 + Dx1[i,:] for i in range(len(Dx1))])
        Dx5 = np.asarray([Dx2 + Dx2[i,:] for i in range(len(Dx2))])
        
        print('calculate new optics function')
        
        Bx1_diff = Bx1 + madx.table.twiss.betx
        Bx2_diff = Bx2 + madx.table.twiss.betx
        Bx3_diff = Bx3 + madx.table.twiss.betx
        Bx4_diff = Bx4 + madx.table.twiss.betx
        Bx5_diff = Bx5 + madx.table.twiss.betx
        
        By1_diff = By1 + madx.table.twiss.bety
        By2_diff = By2 + madx.table.twiss.bety
        By3_diff = By3 + madx.table.twiss.bety
        By4_diff = By4 + madx.table.twiss.bety
        By5_diff = By5 + madx.table.twiss.bety
        
        Dx1_diff = Dx1 + madx.table.twiss.dx
        Dx2_diff = Dx2 + madx.table.twiss.dx
        Dx3_diff = Dx3 + madx.table.twiss.dx
        Dx4_diff = Dx4 + madx.table.twiss.dx
        Dx5_diff = Dx5 + madx.table.twiss.dx
        
        print('calculate lowest std (takes a while)')
        
        idx1, BB1 = find_best_conf(madx.table.twiss.s,Bx1_diff,By1_diff,Dx1_diff)
        idx2, BB2 = find_best_conf(madx.table.twiss.s,Bx2_diff,By2_diff,Dx2_diff)
        idx3, BB3 = find_best_conf(madx.table.twiss.s,Bx3_diff,By3_diff,Dx3_diff)
        idx4, BB4 = find_best_conf(madx.table.twiss.s,Bx4_diff,By4_diff,Dx4_diff)
        idx5, BB5 = find_best_conf(madx.table.twiss.s,Bx5_diff,By5_diff,Dx5_diff)
        
        print('Get madx results of chosen configs')
        
        BB1 = get_madx_mean1(madx,idx1,s_operational,s_possible)
        print(tester_func(madx))
        BB2 = get_madx_mean2(madx,idx2,s_operational,s_possible)
        print(tester_func(madx))
        BB3 = get_madx_mean3(madx,idx3,s_operational,s_possible)
        print(tester_func(madx))
        BB4 = get_madx_mean4(madx,idx4,s_operational,s_possible)
        print(tester_func(madx))
        BB5 = get_madx_mean5(madx,idx5,s_operational,s_possible)
        print(tester_func(madx))
        
        
        if len(s_operational) >= 40:
            BB_list = np.asarray([BB2,BB3,BB5])
        elif len(s_operational) >= 39:
            BB_list = np.asarray([BB1,BB2,BB3,BB5])
        else:
            BB_list = np.asarray([BB1,BB2,BB3,BB4,BB5])
        
        print('Add or remove selected quad')
        
        if BB_list.min() == BB1:
            madx,flag = add_wmarkers(madx,s_operational,s_possible[idx1])
            print('Adding to quad config: section ',s_possible[idx1] +1)
            s_operational = np.append(s_operational,s_possible[idx1])
        if BB_list.min() == BB2:
            madx,flag = remove_wmarkers(madx,s_operational,s_operational[idx2])
            print('Removing to quad config: section ',s_operational[idx2] +1)
            s_operational = np.delete(s_operational, idx2)
        if BB_list.min() == BB3:
            madx,flag = add_wmarkers(madx,s_operational,s_possible[idx3[0]])
            print('Adding to quad config: section ',s_possible[idx3[0]] +1)
            
            madx,flag = remove_wmarkers(madx,s_operational,s_operational[idx3[1]])
            print('Removing to quad config: section ',s_operational[idx3[1]] +1)
            s_operational = np.append(s_operational,s_possible[idx3[0]])
            s_operational = np.delete(s_operational, idx3[1])
        if BB_list.min() == BB4:
            madx,flag = add_wmarkers(madx,s_operational,s_possible[idx4[0]])
            print('Adding to quad config: section ',s_possible[idx4[0]] +1)
            s_operational = np.append(s_operational,s_possible[idx4[0]])
            
            madx,flag = add_wmarkers(madx,s_operational,s_possible[idx4[1]])
            print('Adding to quad config: section ',s_possible[idx4[1]] +1)
            s_operational = np.append(s_operational,s_possible[idx4[1]])
        if BB_list.min() == BB5:
            madx,flag = remove_wmarkers(madx,s_operational,s_operational[idx5[0]])
            print('Removing to quad config: section ',s_operational[idx5[0]] +1)
            
            madx,flag = remove_wmarkers(madx,s_operational,s_operational[idx5[1]])
            print('Removing to quad config: section ',s_operational[idx5[1]] +1)
            s_operational = np.delete(s_operational, idx5[0])
            s_operational = np.delete(s_operational, idx5[1])
                  
        
        temp2 = counter(s_operational)
        print(temp2)
        
        
        actual_std = get_mean_optics(madx)
        if actual_std > std_min:
            print('old std:', std_min)
            print('new std found:', actual_std)
            print('#---------------------------------')
            print(s_operational)
            #break
        else:
            print('old std:', std_min)
            print('new std found:', actual_std)
            print('#---------------------------------')
            std_min = actual_std
            best_config = s_operational
            
        x=x+1
        
        
except RuntimeError:
    print('error')
    
print(best_config)

    




