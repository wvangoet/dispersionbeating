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
        global,sequence=PS,Q1= 6.10;
        global,sequence=PS,Q2= 6.10;
        jacobian, calls = 50000, tolerance=1.0e-15;
        endmatch;
        ''' )
    madx.twiss()
    

    return madx,posz

def make_interpolate_function(pos, beta):
        
        uniQ_list = [0]
        for j in range(len(pos)-1):
            if pos[j] == pos[j+1]:
                pass
            else:
                uniQ_list.append(j+1)
                
        output = interp1d(list(pos[uniQ_list]),list(beta[uniQ_list]),kind='cubic')
    
        return output

def get_mean_optics(madx,alpha):
    linspace = np.linspace(4,628,2000)
    
    gotten_beta1 = make_interpolate_function(madx.table.twiss.S.T,madx.table.twiss.BETX)
    gotten_beta2 = make_interpolate_function(madx.table.twiss.S.T,madx.table.twiss.BETY)
    gotten_disp = make_interpolate_function(madx.table.twiss.S.T,madx.table.twiss.dx)
    
    BetaBeating = np.mean([np.std(gotten_beta1(linspace)),np.std(gotten_beta2(linspace)),alpha*np.std(gotten_disp(linspace))])
                
    return BetaBeating

def get_mean_optics2(s,betx,bety,dx,alpha):
    linspace = np.linspace(4,628,2000)
    
    gotten_beta1 = make_interpolate_function(s,betx)
    gotten_beta2 = make_interpolate_function(s,bety)
    gotten_disp = make_interpolate_function(s,dx)
    
    BetaBeating = np.mean([np.std(gotten_beta1(linspace)),np.std(gotten_beta2(linspace)),alpha*np.std(gotten_disp(linspace))])
                
    return BetaBeating


def get_phase_advance(mux, bx, betx, s):
    
    integral = np.zeros(len(mux))
    for i in range(1,len(s)):
        integral[i] = integral[i-1] + (s[i]-s[i-1])*((bx[i]/(betx[i]**2)) + (bx[i-1]/(betx[i-1]**2)))/2 
        
    phase_advance = mux - integral/(2*np.pi)
    
    return phase_advance

def integrate_bend_from_optics(posz,betx,Q1,mux,angle,L):

    pos_l = []
    for j in range(len(posz)):
        pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(np.abs(betx[posz[j]])))*np.cos(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1)
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


def Calculate_dispersion_from_optics(int_steps,madx,betx,Q1,mux):
    
    pos_bend, neg_bend = get_fdBend(int_steps,madx)
    
    L_F = madx.eval('L_F') 
    L_D = madx.eval('L_D') 
    angle_F = madx.eval('angle_F') 
    angle_D = madx.eval('angle_D') 
        
    Dx_pos_bend = np.sum(np.asarray([integrate_bend_from_optics(i,betx,Q1,mux,angle_F,L_F)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_bend_from_optics(i,betx,Q1,mux,angle_D,L_D)
                    for i in neg_bend]), axis = 0)
    
    dispersion = (np.sqrt(np.abs(betx))/(2*np.sin(np.pi*Q1))) * (Dx_pos_bend + Dx_neg_bend)
    
    
    return dispersion/madx.eval('beam->beta')

def calc_std_from_config(int_steps,alpha,s_operational,s_possible,ssz):
    madx,posz = place_quads_wmarkers(int_steps,s_operational,s_possible,ssz)
    std_min = get_mean_optics(madx,alpha)
    return std_min

def best_changes_to_config(int_steps,alpha,num_confs,ssz,s_operational,s_possible):
    t0_madx = time.time()
    
    madx,posz = place_quads_wmarkers(int_steps,s_operational,s_possible,ssz)
    std_min = get_mean_optics(madx,alpha)
        
    pos_possible = np.asarray(posz)[s_possible]
    pos_operational = np.asarray(posz)[s_operational]
    
    t1_madx = time.time()
    
    t0_beating = time.time()
    
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
    
    add_Bx_diff = [madx.table.twiss.betx + err for err in add_Bx_err]
    add_By_diff = [madx.table.twiss.bety + err for err in add_By_err]
    rem_Bx_diff = [madx.table.twiss.betx + err for err in rem_Bx_err]
    rem_By_diff = [madx.table.twiss.bety + err for err in rem_By_err] 
    
    add_Dx_diff = [Calculate_dispersion_from_optics(int_steps,madx,add_Bx_diff[j],madx.table.summ.q1 + add_delta_Q[j][0],add_mu[j])   for j in range(len(add_Bx_diff))]
    rem_Dx_diff = [Calculate_dispersion_from_optics(int_steps,madx,rem_Bx_diff[j],madx.table.summ.q1 + rem_delta_Q[j][0],rem_mu[j])   for j in range(len(rem_Bx_diff))]

    t1_beating = time.time()
    
    t0_std = time.time()
    
    add_total_std = np.asarray([get_mean_optics2(madx.table.twiss.s.T,add_Bx_diff[i],add_By_diff[i],add_Dx_diff[i],alpha) for i in range(len(add_Bx_err))])
    rem_total_std = np.asarray([get_mean_optics2(madx.table.twiss.s.T,rem_Bx_diff[i],rem_By_diff[i],rem_Dx_diff[i],alpha) for i in range(len(rem_Bx_err))])
    
    t1_std = time.time()
    
    t0_select = time.time()
    
    if len(s_operational)>=50:
        temp_total_std = list(rem_total_std)
        s_for_plot = s_operational
    else:
        temp_total_std = list(np.hstack((add_total_std,rem_total_std)))
        s_for_plot = np.hstack((s_possible,s_operational))
    temp_idx = [temp_total_std.index(np.sort(temp_total_std)[i]) for i in range(num_confs) if np.sort(temp_total_std)[i] <= std_min]
    ssz=np.array(ssz)
    
    t1_select = time.time()
    
    t0_plot = time.time()
    
    plt.plot(ssz[s_possible],add_total_std,'go',ssz[s_operational],rem_total_std,'ro',ssz[s_for_plot[temp_idx]],np.array(temp_total_std)[temp_idx],'k*')
    plt.axhline(y= std_min, linestyle=':', c='k')
    plt.legend(['add quad', 'remove quad'])
    #plt.savefig('img/old_quad_enumeration.png', dpi=300)
    plt.show()
    
    t1_plot = time.time()
    
    print('madx: ',t1_madx-t0_madx,'beating: ',t1_beating-t0_beating,'std: ',t1_std-t0_std,'select: ',t1_select-t0_select,'plot: ',t1_plot-t0_plot)
    
    madx.quit()
    
    result = []
    for idx in temp_idx:
        if len(s_operational)>=50:
            temp_rem_list = list(s_operational)
            temp_rem_list.remove(s_operational[idx])
            result.append(np.array(temp_rem_list))
        elif idx > len(s_possible)-1:
            temp_rem_list = list(s_operational)
            temp_rem_list.remove(s_operational[idx-len(s_possible)])
            result.append(np.array(temp_rem_list))
        else:
            result.append(np.hstack((s_operational,s_possible[idx])))
            
    return result
    
    
    

        
        
            
        
    
            
            
        
        