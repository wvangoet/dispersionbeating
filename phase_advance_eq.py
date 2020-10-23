#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:21:00 2020

@author: wietse
"""

import numpy as np
import pickle
#import pandas
from zoopt import ExpOpt
from scipy.interpolate import interp1d
from cpymad.madx import Madx
from cl2pd import madx as cl2madx

#%%

madx = Madx()
madx.option(echo=False)
madx.input('BEAM, PARTICLE=PROTON, PC = 2.14;')
madx.input('BRHO := BEAM->PC * 3.3356;')
# call sequence of main units
with open('ps_mu.seq', 'r') as main_unit:
    MU_sequence = main_unit.read()
madx.input(MU_sequence)
#call sequence of straight section elements
with open('ps_ss_mod.seq', 'r') as SS:
    SS_sequence = SS.read()
madx.input(SS_sequence)
# call general strength file
with open('ps_mod.str', 'r') as main_str:
    PS_strength = main_str.read()
madx.input(PS_strength)
# call configuration strength file
with open('ps_pro_bare_machine.str', 'r') as strengths:
    strength = strengths.read()
madx.input(strength)

with open('macros.ptc', 'r') as macros:
    macro = macros.read()
madx.input(macro)


with open('remove_elements.seq', 'r') as removess:
    removes = removess.read()

match_string = '''
kf = 0;
kd = 0;

Qx:=0.1;
Qy:=0.1;

use, sequence=PS;
match, use_macro;
    vary, name= pfwk1_f;
    vary, name= pfwk1_d;
    use_macro, name = ptc_twiss_macro(2,0,0);
    constraint, expr = table(ptc_twiss_summary,Q1) = Qx;
    constraint, expr = table(ptc_twiss_summary,Q2) = Qy;
jacobian, calls=50000,bisec=3,tolerance=1e-18;
endmatch;
'''

madx.input('use, sequence=PS;')
madx.input(match_string)
madx.input('twiss, file=pfw_10.txt;')


#%%

def proximity_check(f,a,b,err):
    bool1 = False
    if f % a <= b+err and f % a >= b-err:
        bool1 = True
    return bool1

def lowest_beater(dict1,mus,uneven_bool):
    minim = 0.01
    output= 101
    for i in dict1:
        if i % 2 == int(uneven_bool):
            for j in dict1[i]:
                if np.abs((np.abs(mus[i]-mus[j]) % np.pi) - (np.pi/2)) < minim and np.abs(i-j)<=4:
                    minim= np.abs((np.abs(mus[i]-mus[j]) % np.pi) - (np.pi/2))
                    output = i 

    return output
            


#%%
import matplotlib.pyplot as plt

def find_closest(phases,mu):
    idx = np.argmin(np.abs((phases-mu)))
    return [idx]

s_possible = [ 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 24, 25, 26,
                27, 30, 31, 34, 35, 38, 39, 44, 45, 48, 49, 50, 54, 55, 58, 59, 62,
                64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84, 85, 88, 89, 94, 95, 98,
                99]

f1 = cl2madx.tfs2pd('pfw_25.txt')
Table1=f1.iloc[0].TABLE
table1 = Table1.to_numpy()

ssz = []
idx_ssz = []
delta = [1.486343, 2.886343]
for i,elem in enumerate(table1[1:,0]):
    if elem.startswith('PS') and elem.endswith('START'):
        if int(elem[2:4]) in s_possible:
            print(elem)
            if (elem[3] == '1') or (elem[3] == '6'):
                ssz.append(table1[i,2] + delta[1])
                idx_ssz.append(i)
            else:
                ssz.append(table1[i,2] + delta[0])
                idx_ssz.append(i)
ssz = np.asarray(np.sort(ssz))

phase_advances_x = np.asarray(Table1['MUX'][idx_ssz])*2*np.pi
phase_advances_y = np.asarray(Table1['MUY'][idx_ssz])*2*np.pi
plt.plot([(i - phase_advances_y[1])/(np.pi) % 1 for i in phase_advances_y])
plt.plot([(i - phase_advances_y[1])/(np.pi) % 1 for i in phase_advances_y], 'ro')
plt.plot(np.ones((100,1))*0.5,'k')
plt.show()

j = 0
eq1_dict = {}
eq2_dict = {}
for i,mu1 in enumerate(phase_advances_x):
        eq1_dict[i] = [k for k,mu2 in enumerate(phase_advances_x) if  proximity_check(np.abs(mu1-mu2),np.pi,np.pi/2,0.15)]
        if not eq1_dict[i]:
            eq1_dict[i] = find_closest(phase_advances_x,mu1)
for i,mu1 in enumerate(phase_advances_y):       
        eq2_dict[i] = [k for k,mu2 in enumerate(phase_advances_y) if  proximity_check(np.abs(mu1-mu2),np.pi,np.pi/2,0.15)]
        if not eq2_dict[i]:
            eq2_dict[i] = find_closest(phase_advances_y,mu1)
pos = [0]*4
final_pos = []
uneven_bool = False
while j < 40:
    print('j = ',j)
    minim=1
    minim2 = 15
    if uneven_bool:
        phase_advances = phase_advances_y
        eq_dict = eq2_dict
    else:
        phase_advances = phase_advances_x
        eq_dict = eq1_dict
    #pos2 = eq1_dict[pos1][np.asarray([ np.abs(phase_advances_x[pos1]-phase_advances_x[b]) for b in eq1_dict[pos1]]).argmin()]
    for a in eq_dict:
        if a == pos[0] or not(bool(a%2) == uneven_bool):
            pass
        else:
            for b in eq_dict[a]:
                if b in final_pos or np.abs(a-b)>4:
                    pass
                else:
                    for c in eq_dict[pos[0]]:
                        if a == c or b == c or c in final_pos or np.abs(pos[0]-c)>4:
                            pass
                        else:
                            temp_min = np.abs((((0.25)*np.abs(phase_advances[pos[0]]+phase_advances[c]-phase_advances[a]-phase_advances[b]) % np.pi) - (np.pi/2)))
                            if temp_min < minim and np.std([pos[0],a,b,c]) < minim2:
                                pos[1] = c
                                pos[2] = a
                                pos[3] = b
                                minim2 = np.std(pos)
    for d in range(4):
        print(pos[d])
        final_pos.append(pos[d])
        eq1_dict.pop(pos[d])
        eq2_dict.pop(pos[d])
        j=j+1 
    uneven_bool = (j / 4 % 2== 1)
    #pos[0] = lowest_beater(eq_dict,phase_advances,uneven_bool)
    if uneven_bool == True:
        pos[0] = pos[0] + 1
    else:
        pos[0] = pos[0] + 15
    if pos[0] == 101:
        print("couldn't find good couple")
        print("breaking loop")
        break
    
    
#%%

import get_match_and_beta
import OptEnv as env_mod
from scipy.interpolate import interp1d
from cpymad.madx import Madx
import matplotlib.pyplot as plt
from cl2pd import madx as cl2madx
linspace = np.linspace(4,628,2000)  

                     
#%%
from BB_lib import *

f1 = madx.tfs2pd('pfw_25.txt')
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

s_possible = [ 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 24, 25, 26,
                27, 30, 31, 34, 35, 38, 39, 44, 45, 48, 49, 50, 54, 55, 58, 59, 62,
                64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84, 85, 88, 89, 94, 95, 98,
                99]

taken_pos = [0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16,17,48, 49, 50, 54, 55, 58, 59, 62,
                64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84]

start = 13

pos4 = find_four_adept(start,taken_pos,s_possible,phase_advances_x,phase_advances_y)
    


