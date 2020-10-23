#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:15:12 2020

@author: wvangoet
"""

from Dispersion_Beating_lib import * 

optics_array = []
 
#s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])
s_operational = [1]
for i in s_operational:
#    phase_off = -np.pi/200 *i
    phase_off = 0
    #print(phase_off)
    dk_factor = 1
    optics = check_optics_add_one(i,False,phase_off,dk_factor)
    optics_array.append(optics)
    