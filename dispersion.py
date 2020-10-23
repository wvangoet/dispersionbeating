#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:08:02 2020

@author: wvangoet
"""
import numpy as np

def get_fdBend(int_steps,madx):
    
    pos_bend = []
    neg_bend = []
    j = k = 0
    
    temp_bend1=[]
    temp_bend2=[]
    for i,elem in enumerate(madx.table.twiss.name):
        
        if j >=int_steps:
            j=0
            pos_bend.append(temp_bend1)
            temp_bend1=[]
            
        if k >=int_steps:
            k=0
            neg_bend.append(temp_bend2)
            temp_bend2=[]
            
        if elem.startswith('pr.bh') and elem.endswith('f_den:1'):
            temp_bend1.append(i)
            j=j+1
        
        if elem.startswith('pr.bh') and elem.endswith('d_den:1'):
            temp_bend2.append(i)
            k=k+1
        
        for l in range(1,int_steps-1):
            if elem.startswith('pr.bh') and elem.endswith('d..'+str(l)+':1'):
                temp_bend2.append(i)
                k=k+1
                
            if elem.startswith('pr.bh') and elem.endswith('f..'+str(l)+':1'):
                temp_bend1.append(i)
                j=j+1
                
        if elem.startswith('pr.bh') and elem.endswith('f_dex:1'):
            temp_bend1.append(i)
            j=j+1
        
        if elem.startswith('pr.bh') and elem.endswith('d_dex:1'):
            temp_bend2.append(i)
            k=k+1
                
    return pos_bend, neg_bend
    

def calc_disp(int_steps,madx):
   
    pos_bend, neg_bend = get_fdBend(int_steps,madx)
    
    L_F = madx.eval('L_F') 
    L_D = madx.eval('L_D') 
    angle_F = madx.eval('angle_F') 
    angle_D = madx.eval('angle_D') 
        
    Dx_pos_bend = np.sum(np.asarray([integrate_bend3(i,madx,angle_F,L_F)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_bend3(i,madx,angle_D,L_D)
                    for i in neg_bend]), axis = 0)
    
    Dx_first = (np.sqrt(madx.table.twiss.betx)/(2*np.sin(np.pi*madx.table.summ.q1))) * (Dx_pos_bend + Dx_neg_bend)
               
    return Dx_first

def integrate_bend3(posz,madx,angle,L):

    pos_l = []
    for j in range(len(posz)):
        pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(madx.table.twiss.betx[posz[j]]))*np.cos(2*np.pi*np.abs(madx.table.twiss.mux[posz[j]] - madx.table.twiss.mux) - np.pi*madx.table.summ.q1)
        pos_l.append(pos_temp)
        
    result = L/(3*(len(pos_l)-1)) *np.sum(np.array([ (pos_l[i-1]+4*pos_l[i]+pos_l[i+1]) for i in range(1,len(pos_l)-1,2)]), axis = 0)

    return result


########################################################
########################################################
########################################################
  
    
def get_fdBend_FODO(int_steps,madx):
    
    pos_bend = []
    j = 0
    
    temp_bend1=[]
    for i,elem in enumerate(madx.table.twiss.name):
        
        if j >=int_steps:
            j=0
            pos_bend.append(temp_bend1)
            temp_bend1=[]
            
        if elem.startswith('b') and elem.endswith('_den:1'):
            temp_bend1.append(i)
            j=j+1
        
        for l in range(1,int_steps-1):  
            if elem.startswith('b') and elem.endswith('..'+str(l)+':1'):
                temp_bend1.append(i)
                j=j+1
                
        if elem.startswith('b') and elem.endswith('_dex:1'):
            temp_bend1.append(i)
            j=j+1
                
    return pos_bend

def calc_disp_FODO(int_steps,madx):
   
    pos_bend = get_fdBend_FODO(int_steps,madx)
    
    L_F = madx.eval('dipoleLength') 
    angle_F = madx.eval('myAngle') 
        
    Dx_pos_bend = np.sum(np.asarray([integrate_bend3(i,madx,angle_F,L_F)
                    for i in pos_bend]), axis = 0)
    
    Dx_first = (np.sqrt(madx.table.twiss.betx)/(2*np.sin(np.pi*madx.table.summ.q1))) * (Dx_pos_bend)
               
    return Dx_first


def integrate_bend4(posz, madx,Bx_err,angle,L,pos,dQ, phase_off):
    
    pos_l = []
    for j in range(len(posz)):
        pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(madx.table.twiss.betx[posz[j]] + Bx_err[posz[j]]))*np.cos(2*np.pi*np.abs(madx.table.twiss.mux[posz[j]] - madx.table.twiss.mux + phase_off)  - np.pi*(madx.table.summ.q1 + dQ))
        pos_l.append(pos_temp)
        
    result = L/(3*(len(pos_l)-1)) *np.sum(np.array([ (pos_l[i-1]+4*pos_l[i]+pos_l[i+1]) for i in range(1,len(pos_l)-1,2)]), axis = 0)

    return result

def get_dispersion_beating2_FODO(int_steps,madx,Bx_err,pos, dQ, phase_off):
   
    pos_bend = get_fdBend_FODO(int_steps,madx)
 
    L_F = madx.eval('dipoleLength') 
    angle_F = madx.eval('myAngle') 
        
    Dx_pos_bend = np.sum(np.asarray([integrate_bend4(i,madx,Bx_err,angle_F,L_F, pos,dQ, phase_off)
                    for i in pos_bend]), axis = 0)
    
    Dx_first = Dx_pos_bend
               
    return Dx_first

