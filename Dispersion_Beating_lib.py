import numpy as np
import pickle
import time
#import pandas
from zoopt import ExpOpt
from scipy.interpolate import interp1d
from cpymad.madx import Madx
from cl2pd import madx as cl2madx
import matplotlib.pyplot as plt

def place_quads_wmarkers(int_steps,pos,s_pos,ssz):
    madx = Madx(stdout=False)
    madx.option(echo=False, warn=False, info=False, debug=False, verbose=False)
    madx.input('BEAM, PARTICLE=PROTON, PC = 2.14')
    madx.input('BRHO := BEAM->PC * 3.3356;')
    madx.call(file='ps_mu.seq')
    madx.call(file='ps_ss_mod.seq')
    madx.call(file='ps_50LeQ.str')
    madx.call(file='ps_pro_bare_machine.str')
    
    madx.call(file='remove_elements.seq')
    madx.input('seqedit, sequence = PS;')
    madx.input('select, flag=seqedit, class = MQNAAIAP;')
    madx.input('select, flag=seqedit, class = MQNABIAP;')
    madx.input('select, flag=seqedit, class = MQSAAIAP;')
    madx.input('select, flag=seqedit, class = QNSD;')
    madx.input('select, flag=seqedit, class = QNSF;')
    
    madx.input('use, sequence = PS;')
    madx.input('seqedit,sequence = PS;flatten;endedit;')
    madx.input('seqedit,sequence = PS;remove, element=SELECTED;endedit;')
    madx.input('endedit;')  
    madx.input('use, sequence = PS;')
    madx.input('seqedit, sequence = PS;')
    
    for i in range(100):
        madx.input('MARK%02d: MARKER;' %(i+1))
        madx.input('install, element= MARK%02d, at=' %(i+1) + str(ssz[i])+';')
        
    for i in s_pos:
        madx.input('MARK%02d_2: MARKER;' %(i+1))
        madx.input('install, element= MARK%02d_2, at=' %(i+1) + str(ssz[i]+0.01)+';')
    
    madx.input('endedit;')
    madx.input('use, sequence=PS;')
    madx.input('select, flag=makethin, CLASS=SBEND, THICK= false, SLICE ='+str(int_steps)+';')
    madx.input('makethin, sequence=PS;')
    madx.input('use, sequence=PS;')
    madx.twiss()
    
    posz = [i for i,elem in enumerate(madx.table.twiss.name) if elem.startswith('mark') and not(elem.endswith('_2:1'))]
    
    madx.input('seqedit, sequence = PS;')
    
    for s_idx in pos:
        if s_idx == 99: 
            madx.input('PR.QDN00: MULTIPOLE, KNL:={0,kd};')
            madx.input('replace, element=MARK100, by=PR.QDN00;')
        elif (s_idx % 2) == 1: 
            madx.input('PR.QDN%02d: MULTIPOLE, KNL:={0,kd};' %(s_idx+1))
            madx.input('replace, element=MARK%02d, by=PR.QDN%02d;' %(s_idx+1,s_idx+1))
        else:
            madx.input('PR.QFN%02d: MULTIPOLE, KNL:={0,kf};' %(s_idx+1))
            madx.input('replace, element=MARK%02d, by=PR.QFN%02d;' %(s_idx+1,s_idx+1))
            
    madx.input('endedit;')
    madx.input('use, sequence=PS;')
    madx.input('''
        match, sequence=PS;
        vary, name= kd, step= 0.00001;
        vary, name= kf, step= 0.00001;
        global,sequence=PS,Q1= 6.21;
        global,sequence=PS,Q2= 6.24;
        jacobian, calls = 50000, tolerance=1.0e-15;
        endmatch;
        ''' )
    madx.twiss()
    

    return madx,posz

def remove_wmarkers(madx,s_ope,pos):
    match_flag = False
    madx.input('seqedit, sequence = PS;')
    if pos in counter(s_ope):
        if pos == 99: 
            madx.input('replace, element=PR.QDN00_2, by=MARK100_2;')
        elif (pos % 2) == 1: 
            madx.input('replace, element=PR.QDN%02d_2, by=MARK%02d_2;' %(pos+1,pos+1))
        else:
            madx.input('replace, element=PR.QFN%02d_2, by=MARK%02d_2;' %(pos+1,pos+1))
    else:
        if pos == 99: 
            madx.input('replace, element=PR.QDN00, by=MARK100;')
        elif (pos % 2) == 1: 
            madx.input('replace, element=PR.QDN%02d, by=MARK%02d;' %(pos+1,pos+1))
        else:
            madx.input('replace, element=PR.QFN%02d, by=MARK%02d;' %(pos+1,pos+1))
      
    madx.input('endedit;')
    madx.input('use, sequence=PS;')
      
    madx.twiss()
    if madx.table.summ.Q1 >= 6.15 :
        print('matching failed')
        match_flag = True
    
    return madx, match_flag

def counter(poss):
    poss.sort()
    output = []
    
    prev =-1
    count = 0
    
    for item in poss:
        if item == prev:
            count = count +1
        else:
            count = 1
        prev = item
        
        if count == 2:
            output.append(item)
    return output

def integrate_func_eq_12_1(posz,betx,Bx,Q1,mux,angle,L,quad_pos):

    pos_l = []
    for j in range(len(posz)):
         if posz[j] > quad_pos:
            pos_temp = (2*np.sin(angle/2)/L)*(Bx[posz[j]]/np.sqrt(np.abs(betx[posz[j]])))*np.cos(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1)
         else:
            pos_temp = (2*np.sin(angle/2)/L)*(Bx[posz[j]]/np.sqrt(np.abs(betx[posz[j]])))*np.cos(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1)
         pos_l.append(pos_temp)
        
    result = L/(3*(len(pos_l)-1)) *np.sum(np.array([ (pos_l[i-1]+4*pos_l[i]+pos_l[i+1]) for i in range(1,len(pos_l)-1,2)]), axis = 0)

    return result

def integrate_func_eq_12_2(posz,betx,Mux_err,Bx,Q1,mux,angle,L,quad_pos):

    pos_l = []
    for j in range(len(posz)):
        if posz[j] > quad_pos:
            pos_temp = (2*np.sin(angle/2)/L)*Mux_err[posz[j]]*np.sqrt(np.abs(betx[posz[j]]))*np.sin(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1)
        else:
            pos_temp = (2*np.sin(angle/2)/L)*Mux_err[posz[j]]*np.sqrt(np.abs(betx[posz[j]]))*np.sin(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1)
        pos_l.append(pos_temp)
        
    result = L/(3*(len(pos_l)-1)) *np.sum(np.array([ (pos_l[i-1]+4*pos_l[i]+pos_l[i+1]) for i in range(1,len(pos_l)-1,2)]), axis = 0)

    return result

def integrate_func_eq_21(posz,betx,Q1,mux,angle,L,quad_pos):

    pos_l = []
    for j in range(len(posz)):
        if posz[j] > quad_pos:
            pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(np.abs(betx[posz[j]])))*np.cos(4*np.pi*(mux[quad_pos] - mux) - 2*np.pi*(mux[posz[j]] - mux) - 3*np.pi*Q1)
        else:
            pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(np.abs(betx[posz[j]])))*np.cos(4*np.pi*(mux[quad_pos] - mux) - 2*np.pi*(mux[posz[j]] + Q1 - mux) - 3*np.pi*Q1)
        pos_l.append(pos_temp)
        
    result = L/(3*(len(pos_l)-1)) *np.sum(np.array([ (pos_l[i-1]+4*pos_l[i]+pos_l[i+1]) for i in range(1,len(pos_l)-1,2)]), axis = 0)

    return result

def integrate_func_eq_6(posz,betx,Bx,Mux,dQ1,Q1,mux,angle,L,quad_pos):

    pos_l = []
    #print(posz,quad_pos)
    for j in range(len(posz)):
        if posz[j] > quad_pos:
            #print( mux[posz[j]] - mux )
            pos_temp = (2*np.sin(angle/2)/L)*( (np.sqrt(betx[posz[j]]) + 0.5*(Bx[posz[j]]/np.sqrt(betx[posz[j]])))*np.cos(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1) +
                        np.sqrt(betx[posz[j]])*(2*np.pi*(-Mux[posz[j]] + Mux) + np.pi*dQ1 )*np.sin(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1) )
        else:
            pos_temp = (2*np.sin(angle/2)/L)*( (np.sqrt(betx[posz[j]]) + 0.5*(Bx[posz[j]]/np.sqrt(betx[posz[j]])))*np.cos(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1) +
                        np.sqrt(betx[posz[j]])*(2*np.pi*(-Mux[posz[j]] + Mux) + np.pi*dQ1 )*np.sin(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1) )
        pos_l.append(pos_temp)
        
    result = L/(3*(len(pos_l)-1)) *np.sum(np.array([ (pos_l[i-1]+4*pos_l[i]+pos_l[i+1]) for i in range(1,len(pos_l)-1,2)]), axis = 0)

    return result

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

def Calculate_integral_func_eq_6(int_steps,madx,betx,Q1,mux,Bx,Mux,dQ1,quad_pos):
    
    pos_bend, neg_bend = get_fdBend(int_steps,madx)
    
    L_F = madx.eval('L_F') 
    L_D = madx.eval('L_D') 
    angle_F = madx.eval('angle_F') 
    angle_D = madx.eval('angle_D') 
        
    Dx_pos_bend = np.sum(np.asarray([integrate_func_eq_6(i,betx,Bx,Mux,dQ1,Q1,mux,angle_F,L_F,quad_pos)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_func_eq_6(i,betx,Bx,Mux,dQ1,Q1,mux,angle_D,L_D,quad_pos)
                    for i in neg_bend]), axis = 0)
    
    dispersion = (Dx_pos_bend + Dx_neg_bend)
      
    return dispersion

def Calculate_integral_func_eq_12_1(int_steps,madx,betx,Q1,mux,Bx,quad_pos):
    
    pos_bend, neg_bend = get_fdBend(int_steps,madx)
    
    L_F = madx.eval('L_F') 
    L_D = madx.eval('L_D') 
    angle_F = madx.eval('angle_F') 
    angle_D = madx.eval('angle_D') 
        
    Dx_pos_bend = np.sum(np.asarray([integrate_func_eq_12_1(i,betx,Bx,Q1,mux,angle_F,L_F,quad_pos)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_func_eq_12_1(i,betx,Bx,Q1,mux,angle_D,L_D,quad_pos)
                    for i in neg_bend]), axis = 0)
    
    dispersion = (Dx_pos_bend + Dx_neg_bend)
      
    return dispersion

def Calculate_integral_func_eq_12_2(int_steps,madx,betx,Q1,mux,Bx,Mux_err,quad_pos):
    
    pos_bend, neg_bend = get_fdBend(int_steps,madx)
    
    L_F = madx.eval('L_F') 
    L_D = madx.eval('L_D') 
    angle_F = madx.eval('angle_F') 
    angle_D = madx.eval('angle_D') 
        
    Dx_pos_bend = np.sum(np.asarray([integrate_func_eq_12_2(i,betx,Mux_err,Bx,Q1,mux,angle_F,L_F,quad_pos)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_func_eq_12_2(i,betx,Mux_err,Bx,Q1,mux,angle_D,L_D,quad_pos)
                    for i in neg_bend]), axis = 0)
    
    dispersion = (Dx_pos_bend + Dx_neg_bend)
      
    return dispersion


def Calculate_integral_func_eq_21(int_steps,madx,betx,Q1,mux,quad_pos):
    
    pos_bend, neg_bend = get_fdBend(int_steps,madx)
    
    L_F = madx.eval('L_F') 
    L_D = madx.eval('L_D') 
    angle_F = madx.eval('angle_F') 
    angle_D = madx.eval('angle_D') 
        
    Dx_pos_bend = np.sum(np.asarray([integrate_func_eq_21(i,betx,Q1,mux,angle_F,L_F,quad_pos)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_func_eq_21(i,betx,Q1,mux,angle_D,L_D,quad_pos)
                    for i in neg_bend]), axis = 0)
    
    dispersion = (Dx_pos_bend + Dx_neg_bend)
    
    
    return dispersion

def get_phase_advance(mux, bx, betx, s):
    
    integral = np.zeros(len(mux))
    for i in range(1,len(s)):
        integral[i] = integral[i-1] + (s[i]-s[i-1])*((bx[i]/(betx[i]**2)) + (bx[i-1]/(betx[i-1]**2)))/2 #- (s[i]-s[i-1])*((2*(bx[i]**2)/(betx[i]**3)) + (2*(bx[i]**2)/(betx[i]**3)))/2
        
    phase_advance = - integral/(2*np.pi)
    
    return phase_advance

def get_positions_of_quads():
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
    
    return ssz

def add_wmarkers(madx,s_ope,pos,dk_factor):
    match_flag = False
    madx.input('seqedit, sequence = PS;')
    kd = madx.eval('kd')
    kf = madx.eval('kf')
    
    
    if pos in s_ope:
        if pos == 99: 
            madx.input('PR.QDN00_2: MULTIPOLE, KNL:={0,'+str(kd*dk_factor)+'};')
            madx.input('replace, element=MARK100_2, by=PR.QDN00_2;')
        elif (pos % 2) == 1: 
            madx.input('PR.QDN%02d_2: MULTIPOLE, KNL:={0,'%(pos+1) +str(kd*dk_factor)+'};' )
            madx.input('replace, element=MARK%02d_2, by=PR.QDN%02d_2;' %(pos+1,pos+1))
        else:
            madx.input('PR.QFN%02d_2: MULTIPOLE, KNL:={0,'%(pos+1) +str(kf*dk_factor)+'};')
            madx.input('replace, element=MARK%02d_2, by=PR.QFN%02d_2;' %(pos+1,pos+1))
    else:
        if pos == 99: 
            madx.input('PR.QDN00: MULTIPOLE, KNL:={0,'+str(kd*dk_factor)+'};')
            madx.input('replace, element=MARK100, by=PR.QDN00;')
        elif (pos % 2) == 1: 
            madx.input('PR.QDN%02d: MULTIPOLE, KNL:={0,'%(pos+1) +str(kd*dk_factor)+'};')
            madx.input('replace, element=MARK%02d, by=PR.QDN%02d;' %(pos+1,pos+1))
        else:
            madx.input('PR.QFN%02d: MULTIPOLE, KNL:={0,'%(pos+1) +str(kf*dk_factor)+'};')
            madx.input('replace, element=MARK%02d, by=PR.QFN%02d;' %(pos+1,pos+1))
    madx.input('endedit;')
    madx.input('use, sequence=PS;')
        
    madx.twiss()
    if madx.table.summ.Q1 >= 6.15 :
        print('matching failed')
        match_flag = True
    
    return madx, match_flag

def check_optics_add_one(Position,FODO_flag,phase_off,dk_factor):  #Position is a number 1-100
    
    Position = Position - 1  #easier for python
    
    if Position not in range(100):
        print('Choose a number between 1-100')
        return 0
    
    ssz = get_positions_of_quads() 
    int_step = 15
    
    
    s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
    
    if FODO_flag:
        madx,posz = place_quads_wmarkers_FODO(13)
    else:
        madx,posz = place_quads_wmarkers(int_step,s_operational,range(100),ssz)
        
    pos = posz[Position]
   
    if FODO_flag:
        A=[-madx.eval('kf')*5/140,-madx.eval('kd')*5/140]
        delta_Q = [(-np.arccos(np.cos(2*np.pi*madx.table.summ.q1) - (A[Position%2]/2)*madx.table.twiss.betx[pos]*np.sin(2*np.pi*madx.table.summ.q1))/(2*np.pi)+np.round(madx.table.summ.q1)-madx.table.summ.q1),
            (np.arccos(np.cos(2*np.pi*madx.table.summ.q2) + (A[Position%2]/2)*madx.table.twiss.bety[pos]*np.sin(2*np.pi*madx.table.summ.q2))/(2*np.pi) +np.floor(madx.table.summ.q2)-madx.table.summ.q2)]

   
    else:
        A=[madx.eval('kf')*dk_factor,madx.eval('kd')*dk_factor]
        delta_Q = [(np.arccos(np.cos(2*np.pi*madx.table.summ.q1) - (A[Position%2]/2)*madx.table.twiss.betx[pos]*np.sin(2*np.pi*madx.table.summ.q1))/(2*np.pi)+np.floor(madx.table.summ.q1)-madx.table.summ.q1),
            (np.arccos(np.cos(2*np.pi*madx.table.summ.q2) + (A[Position%2]/2)*madx.table.twiss.bety[pos]*np.sin(2*np.pi*madx.table.summ.q2))/(2*np.pi) +np.floor(madx.table.summ.q2)-madx.table.summ.q2)]
    
    print(A[Position%2],madx.table.twiss.betx[pos], madx.table.summ.q1,madx.eval('kf')  )
    Bx_err = -A[Position%2]/(2*np.sin(2*np.pi*(madx.table.summ.q1 + delta_Q[0])))*madx.table.twiss.betx*madx.table.twiss.betx[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.mux[pos] - madx.table.twiss.mux) -2*np.pi*madx.table.summ.q1)               
    By_err = A[Position%2]/(2*np.sin(2*np.pi*(madx.table.summ.q2 + delta_Q[1])))*madx.table.twiss.bety*madx.table.twiss.bety[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.muy[pos] - madx.table.twiss.muy) -2*np.pi*madx.table.summ.q2) 
    
    Mux_err = np.array(get_phase_advance(madx.table.twiss.mux, Bx_err, madx.table.twiss.betx, madx.table.twiss.s))
    temp_disp = madx.table.twiss.dx*madx.eval('beam->beta')
    temp_ddx = madx.table.twiss.dpx*madx.eval('beam->beta')
    #plt.plot(madx.table.twiss.dpx)
    #plt.plot(temp_disp*(madx.table.twiss.alfx/madx.table.twiss.betx))
    
    #equation 6
#    first_Dx_err = (1/(2*(np.sin(np.pi*madx.table.summ.q1))))* (np.sqrt(madx.table.twiss.betx) + 0.5*(Bx_err/np.sqrt(madx.table.twiss.betx)) - np.pi*delta_Q[0]*(np.sqrt(madx.table.twiss.betx)/np.tan(np.pi*madx.table.summ.q1))) 
#    second_Dx_err = Calculate_integral_func_eq_6(int_step,madx,madx.table.twiss.betx,madx.table.summ.q1,madx.table.twiss.mux,Bx_err,Mux_err,delta_Q[0],pos)
#    print(np.max((1/(2*(np.sin(np.pi*madx.table.summ.q1))))* (np.sqrt(madx.table.twiss.betx) + 0.5*(Bx_err/np.sqrt(madx.table.twiss.betx)) - np.pi*delta_Q[0]*(np.sqrt(madx.table.twiss.betx)/np.tan(np.pi*madx.table.summ.q1)))), np.max(Calculate_integral_func_eq_6(5,madx,madx.table.twiss.betx,madx.table.summ.q1,madx.table.twiss.mux,Bx_err,Mux_err,delta_Q[0],pos)))    
    
    #equation 12
    first_Dx_err = (1/(4*(np.sin(np.pi*madx.table.summ.q1)))) * Calculate_integral_func_eq_12_1(int_step,madx,madx.table.twiss.betx,madx.table.summ.q1,madx.table.twiss.mux,Bx_err,pos)
    second_Dx_err = -(1/(2*(np.sin(np.pi*madx.table.summ.q1)))) * Calculate_integral_func_eq_12_2(int_step,madx,madx.table.twiss.betx,madx.table.summ.q1,madx.table.twiss.mux,Bx_err,Mux_err,pos)
    third_Dx_err =  temp_disp*(0.5*Bx_err/(madx.table.twiss.betx*np.sqrt(madx.table.twiss.betx)) - np.pi*delta_Q[0]/(np.sqrt(madx.table.twiss.betx)*np.tan(np.pi*madx.table.summ.q1)))
    fourth_Dx_err = np.sqrt(madx.table.twiss.betx)*(temp_ddx + temp_disp*(madx.table.twiss.alfx/madx.table.twiss.betx))*(np.pi*delta_Q[0] + 2*np.pi*Mux_err) 
    
    
    #equation 21
#    first_Dx_err = (1/(8*(np.sin(np.pi*madx.table.summ.q1)**2))) * A[Position%2] * madx.table.twiss.betx[pos] * Calculate_integral_func_eq_21(int_step,madx,madx.table.twiss.betx,madx.table.summ.q1,madx.table.twiss.mux,pos)
#    second_Dx_err = ((-1/(4*np.sin(np.pi*madx.table.summ.q1)))* ((temp_disp/np.sqrt(madx.table.twiss.betx))*np.cos(4*np.pi*madx.table.twiss.mux) +  np.sqrt(madx.table.twiss.betx)*( temp_ddx + temp_disp*(madx.table.twiss.alfx/madx.table.twiss.betx))*np.sin(4*np.pi*madx.table.twiss.mux) ) ) * A[Position%2] * madx.table.twiss.betx[pos] *np.cos(2*np.pi*(2*madx.table.twiss.mux[pos] - madx.table.summ.q1))  
#    third_Dx_err =  ((-1/(4*np.sin(np.pi*madx.table.summ.q1)))* ((temp_disp/np.sqrt(madx.table.twiss.betx))*np.sin(4*np.pi*madx.table.twiss.mux) +  np.sqrt(madx.table.twiss.betx)*( temp_ddx + temp_disp*(madx.table.twiss.alfx/madx.table.twiss.betx))*(1-np.cos(4*np.pi*madx.table.twiss.mux) ) ) ) * A[Position%2] * madx.table.twiss.betx[pos] *np.sin(2*np.pi*(2*madx.table.twiss.mux[pos] - madx.table.summ.q1))
#    fourth_Dx_err = -0.25*( np.sqrt(madx.table.twiss.betx)*( temp_ddx + temp_disp*(madx.table.twiss.alfx/madx.table.twiss.betx)) - (temp_disp/(np.tan(np.pi*madx.table.summ.q1)*np.sqrt(madx.table.twiss.betx)) ) )*  A[Position%2] * madx.table.twiss.betx[pos]
    
        
    Full_Dx_err =  (first_Dx_err + second_Dx_err + third_Dx_err + fourth_Dx_err)*np.sqrt(madx.table.twiss.betx)
    #Full_Dx_err =  first_Dx_err*second_Dx_err -temp_disp
    
    temp_betx = madx.table.twiss.betx
    temp_mux = madx.table.twiss.mux
    temp_bety = madx.table.twiss.bety
    #temp_disp = madx.table.twiss.dx*madx.eval('beam->beta')
    temp_Q = madx.table.summ.q1
    temp_Qy = madx.table.summ.q2  
    
#    plt.figure()
#    plt.plot(np.sqrt(temp_betx + Bx_err),'r',np.sqrt(temp_betx)+0.5*(Bx_err/np.sqrt(temp_betx)),'b')
#    plt.show()
#    
#    print((1/np.sin(np.pi*(temp_Q+delta_Q[0]))),(1/np.sin(np.pi*temp_Q))*(1-np.pi*delta_Q[0]*(1/np.tan(np.pi*temp_Q))))
#    
#    plt.figure()
#    plt.plot(np.cos(2*np.pi*np.abs(temp_mux[2000] + Mux_err[2000] - temp_mux - Mux_err) - np.pi*(temp_Q+delta_Q[0]) ),'r',np.cos(2*np.pi*np.abs(temp_mux[2000] - temp_mux ) - np.pi*(temp_Q)) + (-Mux_err[2000] + Mux_err + np.pi*delta_Q[0])*np.sin(2*np.pi*np.abs(temp_mux[2000] - temp_mux ) - np.pi*(temp_Q)),'b')
#    plt.show()
#        
    
    if FODO_flag:
        madx,flag = add_wmarkers_FODO(madx,Position)
    else:        
        madx,flag = add_wmarkers(madx,s_operational,Position,dk_factor)
        #madx,flag = remove_wmarkers(madx,s_operational,Position)
    
    
    temp_betx2 = madx.table.twiss.betx
    temp_mux2 = madx.table.twiss.mux
    temp_bety2 = madx.table.twiss.bety
    temp_disp2 = madx.table.twiss.dx*madx.eval('beam->beta')
    temp_Q2 = madx.table.summ.q1
    temp_Q2y = madx.table.summ.q2
    
#    plt.plot(madx.table.twiss.s,temp_mux2-temp_mux,'r',madx.table.twiss.s,Mux_err,'b')
#    plt.show()
    
    print('quad location at ', madx.table.twiss.s[pos])
    print('madx tune diff:',temp_Q2-temp_Q,' calculated tune diff', delta_Q[0], ' dk = ',A[Position%2] )
    print('madx tune diff:',temp_Q2y-temp_Qy,' calculated tune diff', delta_Q[1], ' dk = ',A[Position%2] )
    
    #print(ssz,madx.table.twiss.s[posz])
    
    plt.figure(0)
    plt.plot(madx.table.twiss.s,temp_betx2-temp_betx,'r',madx.table.twiss.s,Bx_err,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_betx2[posz[Position]]-temp_betx[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1)+', dk = kf*'+str(dk_factor))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta \beta_x (m)$')
    plt.legend(['madx','eqs'])
    plt.savefig('img/dispersion_beating/betx_'+str(Position+1)+str(dk_factor*3)+'.png',dpi=300)
    plt.show()
#       
    plt.figure(1)
    plt.plot(madx.table.twiss.s,temp_bety2-temp_bety,'r',madx.table.twiss.s,By_err,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_bety2[posz[Position]]-temp_bety[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta \beta_y (m)$')
    plt.legend(['madx','eqs'])
    plt.savefig('img/dispersion_beating/bety_'+str(Position+1)+'.png',dpi=300)
    plt.show()
    
    
    plt.figure(3)
    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,Full_Dx_err,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1)+', dk = kf*'+str(dk_factor))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta D_x (m)$')
    plt.legend(['madx','full eqs'])
    #plt.savefig('img/dispersion_beating/phase/full_dispx_'+str(Position+1)+str(int(-phase_off*200/np.pi))+'.png',dpi=300)
    plt.savefig('img/dispersion_beating/full_dispx_'+str(Position+1)+'.png',dpi=300)
    plt.show()
    
    plt.figure(4)
    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,first_Dx_err,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta D_x (m)$')
    plt.legend(['madx','first term'])
    #plt.savefig('img/dispersion_beating/phase/first_dispx_'+str(Position+1)+'.png',dpi=300)
    plt.show()
#    
    plt.figure(5)
    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,second_Dx_err,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta D_x (m)$')
    plt.legend(['madx','second term'])
    #plt.savefig('img/dispersion_beating/phase/second_dispx_'+str(Position+1)+'.png',dpi=300)
    plt.show()
#    
    plt.figure(6)
    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,third_Dx_err,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta D_x (m)$')
    plt.legend(['madx','third term'])
    #plt.savefig('img/dispersion_beating/dispersion_terms/third_dispx_'+str(Position+1)+'.png',dpi=300)
    plt.show()
    
    plt.figure(7)
    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,fourth_Dx_err,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta D_x (m)$')
    plt.legend(['madx','fourth term'])
    #plt.savefig('img/dispersion_beating/dispersion_terms/fourth_dispx_'+str(Position+1)+'.png',dpi=300)
    plt.show()
    
    
    return Bx_err, By_err, Full_Dx_err, delta_Q[0]
    
    

        
        
            
        
    
            
            
        
        