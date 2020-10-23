import numpy as np
import pickle
#import pandas
from zoopt import ExpOpt
from scipy.interpolate import interp1d
from cpymad.madx import Madx
from cl2pd import madx as cl2madx
import collections
from BB_lib import *
from BB_lib2 import Calculate_dispersion_from_optics

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


#s_possible = np.asarray([ 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 24, 25, 26,
#                27, 30, 31, 34, 35, 38, 39, 44, 45, 48, 49, 50, 54, 55, 58, 59, 62,
#                64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84, 85, 88, 89, 94, 95, 98,
#                99])
#    
##s_possible = np.asarray([0])
#
#s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
##s_operational = np.asarray([5])-1
bests_configs = []
bests_stds = []

last_operation = 200
int_steps = 5
#alpha_list=np.array(range(0,11,2))/10
alpha=1
#for alpha in alpha_list:
    
s_possible = np.asarray([ 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 24, 25, 26,
            27, 30, 31, 34, 35, 38, 39, 44, 45, 48, 49, 50, 54, 55, 58, 59, 62,
            64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84, 85, 88, 89, 94, 95, 98,
            99])
    
#s_possible = range(100)

s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
#s_operational = np.asarray([[i,i+1] for i in range(0,100,4)]).reshape((50,)) 
#s_operational =np.array([ 0,  1,  4,  5,  8,  9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32,
#       33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 58, 61, 64, 65,
#       68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97])

madx,posz = place_quads_wmarkers(int_steps,s_operational,s_possible,ssz)

std_min = get_mean_optics(madx,alpha)
std_list = [std_min]
ZZ = 0
try:
    while 1:
        
        quick_match(madx)
        
        beta = madx.eval('beam->beta')
        
        pos_possible = np.asarray(posz)[s_possible]
        pos_operational = np.asarray(posz)[s_operational]
        
        A=[madx.eval('kf'),madx.eval('kd')]
        add_delta_Q = [[(np.arccos(np.cos(2*np.pi*madx.table.summ.q1) - (A[s_possible[i]%2]/2)*madx.table.twiss.betx[pos]*np.sin(2*np.pi*madx.table.summ.q1))/(2*np.pi)+np.floor(madx.table.summ.q1)-madx.table.summ.q1),
            (np.arccos(np.cos(2*np.pi*madx.table.summ.q2) + (A[s_possible[i]%2]/2)*madx.table.twiss.bety[pos]*np.sin(2*np.pi*madx.table.summ.q2))/(2*np.pi) +np.floor(madx.table.summ.q2)-madx.table.summ.q2)] for i,pos in enumerate(pos_possible)]
        
        rem_delta_Q = [[(np.arccos(np.cos(2*np.pi*madx.table.summ.q1) - (-A[s_operational[i]%2]/2)*madx.table.twiss.betx[pos]*np.sin(2*np.pi*madx.table.summ.q1))/(2*np.pi)+np.floor(madx.table.summ.q1)-madx.table.summ.q1),
            (np.arccos(np.cos(2*np.pi*madx.table.summ.q2) + (-A[s_operational[i]%2]/2)*madx.table.twiss.bety[pos]*np.sin(2*np.pi*madx.table.summ.q2))/(2*np.pi)+np.floor(madx.table.summ.q1)-madx.table.summ.q2)] for i,pos in enumerate(pos_operational)]
        
        add_Bx_err = [-A[s_possible[i]%2]/(2*np.sin(2*np.pi*(madx.table.summ.q1)))*madx.table.twiss.betx*madx.table.twiss.betx[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.mux[pos] - madx.table.twiss.mux) -2*np.pi*madx.table.summ.q1) for i,pos in enumerate(pos_possible)]                
        add_By_err = [A[s_possible[i]%2]/(2*np.sin(2*np.pi*(madx.table.summ.q2)))*madx.table.twiss.bety*madx.table.twiss.bety[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.muy[pos] - madx.table.twiss.muy) -2*np.pi*madx.table.summ.q2) for i,pos in enumerate(pos_possible)]
        add_mu = [get_phase_advance(madx.table.twiss.mux, Bx, madx.table.twiss.betx, madx.table.twiss.s) for Bx in add_Bx_err]
        
        rem_Bx_err = [(A[s_operational[i]%2]/(2*np.sin(2*np.pi*(madx.table.summ.q1)))*madx.table.twiss.betx*madx.table.twiss.betx[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.mux[pos] - madx.table.twiss.mux) -2*np.pi*madx.table.summ.q1)) for i,pos in enumerate(pos_operational)]              
        rem_By_err = [(-A[s_operational[i]%2]/(2*np.sin(2*np.pi*(madx.table.summ.q2)))*madx.table.twiss.bety*madx.table.twiss.bety[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.muy[pos] - madx.table.twiss.muy) -2*np.pi*madx.table.summ.q2)) for i,pos in enumerate(pos_operational)]
        rem_mu = [get_phase_advance(madx.table.twiss.mux, Bx, madx.table.twiss.betx, madx.table.twiss.s) for Bx in rem_Bx_err]
        
        add_Dx_err = [Calculate_dispersion_from_optics(int_steps,madx,madx.table.twiss.betx + add_Bx_err[j],madx.table.summ.q1 + add_delta_Q[j][0],add_mu[j])  for j in range(len(add_Bx_err))]
        rem_Dx_err = [Calculate_dispersion_from_optics(int_steps,madx,madx.table.twiss.betx + rem_Bx_err[j],madx.table.summ.q1 + rem_delta_Q[j][0],rem_mu[j])  for j in range(len(rem_Bx_err))]
        
        add_Bx_diff = [madx.table.twiss.betx + err for err in add_Bx_err]
        add_By_diff = [madx.table.twiss.bety + err for err in add_By_err] 
        add_Dx_diff = add_Dx_err
        rem_Bx_diff = [madx.table.twiss.betx + err for err in rem_Bx_err]
        rem_By_diff = [madx.table.twiss.bety + err for err in rem_By_err] 
        rem_Dx_diff = rem_Dx_err
        
        add_total_std = np.asarray([get_mean_optics2(madx.table.twiss.s.T,add_Bx_diff[i],add_By_diff[i],add_Dx_diff[i],alpha) for i in range(len(add_Bx_err))])
        rem_total_std = np.asarray([get_mean_optics2(madx.table.twiss.s.T,rem_Bx_diff[i],rem_By_diff[i],rem_Dx_diff[i],alpha) for i in range(len(rem_Bx_err))])
        
        add_idx = add_total_std.argmin()
        rem_idx = rem_total_std.argmin()
        
        plt.plot(ssz[s_possible],add_total_std,'go',ssz[s_operational],rem_total_std,'ro',ssz[s_possible[add_idx]],add_total_std[add_idx],'k*',ssz[s_operational[rem_idx]],rem_total_std[rem_idx],'k*')
        plt.axhline(y= std_min, linestyle=':', c='k')
        plt.legend(['add quad', 'remove quad'])
        plt.savefig('img/old_quad_enumeration'+str(ZZ)+'.png', dpi=300)
        plt.show()
        
        ZZ=ZZ+1
        
        
        if rem_total_std[rem_idx] <= add_total_std[add_idx]:# or len(s_operational)==40:
            print('REMOVING ...', s_operational[rem_idx]+1)
            madx,flag = remove_wmarkers(madx,s_operational,s_operational[rem_idx])
            temp3 = s_operational[rem_idx]
            s_operational = np.delete(s_operational, rem_idx)  
            guessed_std = rem_total_std[rem_idx]
            
            print(madx.table.summ.q1, 6.1 + rem_delta_Q[rem_idx][0])
            print(madx.table.summ.q2, 6.1 + rem_delta_Q[rem_idx][1])          
            plt.plot(madx.table.twiss.s,madx.table.twiss.betx,'r',madx.table.twiss.s,rem_Bx_diff[rem_idx],'b')
            plt.show()
            plt.plot(madx.table.twiss.s,madx.table.twiss.bety,'r',madx.table.twiss.s,rem_By_diff[rem_idx],'b')
            plt.show()
            plt.plot(madx.table.twiss.s,madx.table.twiss.dx,'r',madx.table.twiss.s,rem_Dx_diff[rem_idx],'b')
            plt.show()
        else:
            print('ADDING ...',s_possible[add_idx]+1)
            madx,flag = add_wmarkers(madx,s_operational,s_possible[add_idx],1)
            temp3 = s_possible[add_idx]
            s_operational = np.append(s_operational,s_possible[add_idx])
            guessed_std = add_total_std[add_idx]
            
            print(madx.table.summ.q1, 6.1 + add_delta_Q[add_idx][0])
            print(madx.table.summ.q2, 6.1 + add_delta_Q[add_idx][1])
            plt.plot(madx.table.twiss.s,madx.table.twiss.betx,'r',madx.table.twiss.s,add_Bx_diff[add_idx],'b')
            plt.show()
            plt.plot(madx.table.twiss.s,madx.table.twiss.bety,'r',madx.table.twiss.s,add_By_diff[add_idx],'b')
            plt.show()
            plt.plot(madx.table.twiss.s,madx.table.twiss.dx,'r',madx.table.twiss.s,add_Dx_diff[add_idx],'b')
            plt.show()
            

        temp2 = counter(s_operational)
        print(temp2)
        
        
        actual_std = get_mean_optics(madx,alpha)
        print('guessed std: ',guessed_std,'    actual std: ', actual_std)        
        if actual_std > std_min:
            print('old std:', std_min)
            print('new std found:', actual_std)
            print('#---------------------------------')
            print(s_operational)
            #break
        else:
            print('sector added or removed:', temp3+1)
            print('old std:', std_min)
            print('new std found:', actual_std)
            print('#---------------------------------')
            std_min = actual_std
            std_list.append(std_min)
            best_config = s_operational
            
        if last_operation == temp3:
            break
        else:
            last_operation = temp3    
except ValueError:
    print('error')
    
bests_configs.append(best_config)
bests_stds.append(std_min)
    
#%%
from BB_lib import * 

optics_array = []
 
s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])
for i in s_operational:
#    phase_off = -np.pi/200 *i
    phase_off = 0
    #print(phase_off)
    dk_factor = 1
    optics = check_optics_add_one(i,False,phase_off,dk_factor)
    optics_array.append(optics)
    

#%%
import matplotlib.pyplot as plt 
  
plt.plot(np.array(optics_array[0][0])*3,'r',optics_array[2][0],'b')
plt.show()
plt.plot(np.array(optics_array[0][1])*3,'r',optics_array[2][1],'b')
plt.show()
plt.plot(np.array(optics_array[0][2])*3,'r',optics_array[2][2],'b')
plt.show()

    
    

#%%
from BB_lib import * 
import matplotlib.pyplot as plt 

disp_diff = []
Quad = 5
x = range(7,9,2)
for i in x:
    dispersion, dD = check_dispersion(Quad,i,False)
    disp_diff.append(dD)

plt.plot(x,-1*np.array(disp_diff))
plt.xticks(x)
plt.xlabel('integration steps')
plt.ylabel(r'$mean(D_{madx}-D_{calc})$')
plt.tight_layout()
plt.savefig('img/dispersion_beating/int_steps_fodo.png')
plt.show()

#%%
from BB_lib import * 
from BB_lib2 import Calculate_dispersion_from_optics

ssz = get_positions_of_quads() 
s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
int_steps = 5
Quad = 5

madx,posz = place_quads_wmarkers(int_steps,s_operational,range(100),ssz)

optics = check_optics_add_one(Quad,False,0,1)

Betx1 = madx.table.twiss.betx 
Q1 = madx.table.summ.q1 
Mux1 = get_phase_advance(madx.table.twiss.mux, optics[0], madx.table.twiss.betx, madx.table.twiss.s)


dispersion_madx1 = madx.table.twiss.dx*madx.eval('beam->beta')
dispersion_calc1 = Calculate_dispersion_from_optics(int_steps,madx,Betx1+optics[0],Q1+optics[3],Mux1)

madx,flag = add_wmarkers(madx,s_operational,Quad-1,1)

Betx2 = madx.table.twiss.betx
Q2 = madx.table.summ.q1 
Mux2 = madx.table.twiss.mux

dispersion_calc2 = Calculate_dispersion_from_optics(int_steps,madx,Betx1+optics[0],Q1+optics[3],Mux2)
dispersion_madx2 = madx.table.twiss.dx*madx.eval('beam->beta')

plt.plot(madx.table.twiss.s, dispersion_calc1,'b',madx.table.twiss.s, dispersion_calc2, 'g', madx.table.twiss.s, dispersion_madx2, 'r')
plt.title(r'$D_x$, section '+str(Quad))
plt.legend([r'$(\beta_{x;0} + \Delta \beta, Q_{x;0} + \Delta Q, \mu_{x;0})$',r'$(\beta_{x;0} + \Delta \beta, Q_{x;0} + \Delta Q, \mu_{x;ERR})$' ,'madx'])
#plt.savefig('img/dispersion_beating/Dispersion_eq_'+str(Quad)+'.png',dpi=300)
plt.show()

#plt.plot(madx.table.twiss.s, Mux1,'b',madx.table.twiss.s, Mux2,'r')
#plt.title(r'$D_x$, section '+str(Quad))
#plt.legend([r'$(\beta_{x;0} + \Delta \beta, Q_{x;0} + \Delta Q, \mu_{x;0})$',r'$(\beta_{x;0} + \Delta \beta, Q_{x;0} + \Delta Q, \mu_{x;ERR})$' ,'madx'])
#plt.savefig('img/dispersion_beating/Dispersion_eq_'+str(Quad)+'.png',dpi=300)
#plt.show()


#%%
import numpy as np
import pickle
import time
#import pandas
from cpymad.madx import Madx
from cl2pd import madx as cl2madx
from BB_lib import *
from find_best_config_lib import best_changes_to_config

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

alpha=1
int_steps=5
num_confs=num_colors=5  
s_possible = np.asarray([ 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 24, 25, 26,
            27, 30, 31, 34, 35, 38, 39, 44, 45, 48, 49, 50, 54, 55, 58, 59, 62,
            64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84, 85, 88, 89, 94, 95, 98,
            99])
#s_possible = range(1)
s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
#s_operational = np.asarray([[i,i+1] for i in range(0,100,4)]).reshape((50,))
#s_operational = np.array([ 0,  1,  4,  5,  8,  9, 12, 13, 16, 17,18, 20, 21, 24, 25, 28, 29, 32,
#       33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 64, 65,
#       68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97,98])
full_config_dict= {}
config_dict = {'0':s_operational}

try:
    while 1:
        print(config_dict.keys())
        t0 = time.time()
        key_list = list(config_dict.keys())
        if key_list == [] or num_confs == 0:
            print('complete')
            break
        for key in key_list: 
            result = best_changes_to_config(int_steps,alpha,num_confs,ssz,config_dict[key],s_possible)
            for i in range(len(result)):
                config_dict[str(key)+str(i)] = result[i]
            
            full_config_dict[key] = config_dict[key]
            element = config_dict.pop(key) 
            
        num_confs = num_confs-1
        t1 = time.time()
        
        print('time = ',t1-t0)
        

except ValueError:
    print('error')

#%%
from find_best_config_lib import calc_std_from_config

max_len = branch_depth = 5

std_dict = {}
for key in full_config_dict:
    std = calc_std_from_config(int_steps,alpha,full_config_dict[key],s_possible,ssz)
    std_dict[key] = std
    print(key, std)
    
with open('config_old_10_branches.pkl', 'wb') as f:
    pickle.dump([std_dict,full_config_dict], f)

#%%
import matplotlib
evenly_spaced_interval = np.linspace(0, 1, num_colors)
colors = [matplotlib.cm.brg(x) for x in evenly_spaced_interval]
max_len = branch_depth = 5

temp1_config_dict = full_config_dict.copy()
temp2_config_dict = full_config_dict.copy()

while max_len>=2:
    for key in temp1_config_dict:    
        if len(key) == max_len:
            key_list = [key[:(i+1)] for i in range(1,max_len)]
            plt.plot(range(max_len), np.hstack((std_dict['0'],[std_dict[key1] for key1 in key_list])),color = colors[int(key[1])])
            
            temp2_config_dict.pop(key)
                
    temp1_config_dict = temp2_config_dict.copy()
    max_len = max_len-1
    
plt.title(r'branching paths, $\alpha$ = '+str(alpha))
plt.ylabel(r'benchmark value ($\alpha$)')
plt.xlabel('branching depth')
plt.xticks(range(branch_depth))
lines = []
for j in range(len(colors)):
    line = matplotlib.lines.Line2D([], [], color=colors[j],label='branch '+str(j+1))
    lines.append(line)
    
plt.legend(handles=lines)
                
    
            
        
            
           
            







