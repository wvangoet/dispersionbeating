import numpy as np
import pickle
#import pandas
from zoopt import ExpOpt
from scipy.interpolate import interp1d
from cpymad.madx import Madx
from cl2pd import madx as cl2madx
import matplotlib.pyplot as plt
from dispersion import *

def proximity_check(f,a,b,err):
    bool1 = False
    if f % a <= b+err and f % a >= b-err:
        bool1 = True
    return bool1

def find_closest(phases,mu,i):
    idx = np.argmin(np.abs((phases-mu)-(np.pi/2)))
    return [idx]

def find_four_adept(start,taken_pos,s_possible,phase_advances_x,phase_advances_y,limit):
        
    current_pos = [k for k in s_possible if not(k in taken_pos)]
    
    phase_advances_x = phase_advances_x[current_pos]
    phase_advances_y = phase_advances_y[current_pos]
    
    
    
    eq1_dict = {}
    eq2_dict = {}
    for i,mu1 in enumerate(phase_advances_x):
            eq1_dict[i] = [k for k,mu2 in enumerate(phase_advances_x) if  proximity_check(np.abs(mu1-mu2),np.pi,np.pi/2,limit)]
            if not eq1_dict[i]:
                eq1_dict[i] = find_closest(phase_advances_x,mu1,i)
    for i,mu1 in enumerate(phase_advances_y):  
            eq2_dict[i] = [k for k,mu2 in enumerate(phase_advances_y) if  proximity_check(np.abs(mu1-mu2),np.pi,np.pi/2,limit)]
            if not eq2_dict[i]:
                eq2_dict[i] = find_closest(phase_advances_y,mu1,i)
    
    if start in current_pos:
        pos = [i for i,j in enumerate(current_pos) if j == start]*4
    else:
        print('idk')
    uneven_bool = bool(start % 2) 

    minim=limit*10
    minim2 = limit*100
    if uneven_bool:
        phase_advances = phase_advances_y
        eq_dict = eq2_dict
    else:
        phase_advances = phase_advances_x
        eq_dict = eq1_dict
    #pos2 = eq1_dict[pos1][np.asarray([ np.abs(phase_advances_x[pos1]-phase_advances_x[b]) for b in eq1_dict[pos1]]).argmin()]
    for a in eq_dict:
        if a == pos[0]:
            pass
        else:
            for b in eq_dict[a]:
                if  np.abs(a-b)>8 or a == b:
                    pass
                else:
                    for c in eq_dict[pos[0]]:
                        if b == c or np.abs(pos[0]-c)>8 or pos[0]==c:
                            pass
                        else:
                            if bool(pos[0]%2):
                                sign_0 = -1
                            else:
                                sign_0 = 1
                                
                            if bool(a%2):
                                sign_a = -1
                            else:
                                sign_a = 1
                            
                            #temp_min = np.abs(np.cos(0.25*(phase_advances[pos[0]]+phase_advances[c]-phase_advances[a]-phase_advances[b])))
                            temp_min = np.abs(np.abs(0.25*(phase_advances[pos[0]]+phase_advances[c]-phase_advances[a]-phase_advances[b])) - np.pi/2)
                            # if sign_0*np.sign(np.cos(0.5*(phase_advances[pos[0]]-phase_advances[c]))) == sign_a*np.sign(np.cos(0.5*(phase_advances[a]-phase_advances[b]))):
                            #     temp_min = np.abs(np.abs(0.25*(phase_advances[pos[0]]+phase_advances[c]-phase_advances[a]-phase_advances[b])) - np.pi/2)
                            # else:
                            #     temp_min = np.abs(np.abs(0.25*(phase_advances[pos[0]]+phase_advances[c]-phase_advances[a]-phase_advances[b])) - np.pi)
                            
                            if temp_min < minim:
                                pos[1] = c
                                pos[2] = a
                                pos[3] = b
                                minim = temp_min
                                
                            # if temp_min < limit and np.std([current_pos[pos[0]],current_pos[a],current_pos[b],current_pos[c]]) < minim2:
                            #     pos[1] = c
                            #     pos[2] = a
                            #     pos[3] = b
                            #     minim = temp_min
                            #     minim2 = np.std([current_pos[j] for j in pos])
                            # elif temp_min < minim and minim > limit:
                            #     pos[1] = c
                            #     pos[2] = a
                            #     pos[3] = b
                            #     minim = temp_min
                            #     minim2 = np.std([current_pos[j] for j in pos])
                                

    if len(pos) != len(set(pos)):
        return [current_pos[j] for j in pos],1
    else:
        disp_effect = minim
        return [current_pos[j] for j in pos], disp_effect

def find_four(start,taken_pos,s_possible,phase_advances_x,phase_advances_y):
    
    eq1_dict = {}
    eq2_dict = {}
    for i,mu1 in enumerate(phase_advances_x):
            eq1_dict[i] = [k for k,mu2 in enumerate(phase_advances_x) if  proximity_check(np.abs(mu1-mu2),np.pi,np.pi/2,0.15)]
    for i,mu1 in enumerate(phase_advances_y):       
            eq2_dict[i] = [k for k,mu2 in enumerate(phase_advances_y) if  proximity_check(np.abs(mu1-mu2),np.pi,np.pi/2,0.15)]
    
    pos = [start]*4
    uneven_bool = bool(start % 2) 

    minim=0.15
    minim2 = 15
    if uneven_bool:
        phase_advances = phase_advances_y
        eq_dict = eq2_dict
    else:
        phase_advances = phase_advances_x
        eq_dict = eq1_dict
    #pos2 = eq1_dict[pos1][np.asarray([ np.abs(phase_advances_x[pos1]-phase_advances_x[b]) for b in eq1_dict[pos1]]).argmin()]
    for a in eq_dict:
        if a == pos[0] or not(bool(a%2) == uneven_bool) or a in taken_pos:
            pass
        else:
            for b in eq_dict[a]:
                if  np.abs(a-b)>4 or b in taken_pos:
                    pass
                else:
                    for c in eq_dict[pos[0]]:
                        if a == c or b == c or np.abs(pos[0]-c)>4 or c in taken_pos:
                            pass
                        else:
                            temp_min = np.abs((((0.25)*np.abs(phase_advances[pos[0]]+phase_advances[c]-phase_advances[a]-phase_advances[b]) % np.pi) - (np.pi/2)))
                            if temp_min < minim and np.std([pos[0],a,b,c]) < minim2:
                                pos[1] = c
                                pos[2] = a
                                pos[3] = b
                                minim2 = np.std(pos)
    # for d in range(4):
    #     print(pos[d])

    return pos


def place_quads(pos,ssz):
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
    
    for s_idx in pos:
        if s_idx == 99: 
            madx.input('PR.QDN00: MULTIPOLE, KNL:={0,kd};')
            madx.input('install, element=PR.QDN00, at=' +str(623.834315)+';')
        elif (s_idx % 2) == 1: 
            madx.input('PR.QDN%02d: MULTIPOLE, KNL:={0,kd};' %(s_idx+1))
            madx.input('install, element=PR.QDN%02d, at=' %(s_idx+1) +str(ssz[s_idx])+';')
        else:
            madx.input('PR.QFN%02d: MULTIPOLE, KNL:={0,kf};' %(s_idx+1))
            madx.input('install, element=PR.QFN%02d, at=' %(s_idx+1) +str(ssz[s_idx])+';')
            
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

    return madx

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

def place_quads_wmarkers_FODO(int_steps):
    madx = Madx(stdout=False)
    madx.option(echo=False, warn=False, info=False, debug=False, verbose=False)
    #madx.input('beam, particle=proton, energy=7000;')
    madx.input('BEAM, PARTICLE=PROTON, PC = 2.14')
    madx.input('BRHO := BEAM->PC * 3.3356;')
    madx.call(file='fodo_ring.seq')
    madx.input('seqedit,sequence = FODO_ring;flatten;endedit;')
    madx.input('use, sequence=FODO_ring;')
    madx.input('select, flag=makethin, CLASS=SBEND, THICK= false, SLICE ='+str(int_steps-2)+';')
    madx.input('makethin, sequence=FODO_ring;')
    madx.input('use, sequence=FODO_ring;')
    madx.twiss()
    
    posz = [i for i,elem in enumerate(madx.table.twiss.name) if elem.startswith('q')]
    

    return madx,posz

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
#    madx.input('''
#        match, sequence=PS;
#        vary, name= kd, step= 0.00001;
#        vary, name= kf, step= 0.00001;
#        global,sequence=PS,Q1= 6.10;
#        global,sequence=PS,Q2= 6.10;
#        jacobian, calls = 50000, tolerance=1.0e-15;
#        endmatch;
#        ''' )
        
    madx.twiss()
    if madx.table.summ.Q1 >= 6.15 :
        print('matching failed')
        match_flag = True
    
    return madx, match_flag

def add_wmarkers_FODO(madx,pos):
    match_flag = False
    madx.input('seqedit, sequence = FODO_ring;')
    
    if (pos % 2) == 1:
        madx.input('q%01d2_new: quadrupole,L=quadrupoleLength, K1= -2.78/cellLength/quadrupoleLength;' %(np.floor(pos/2)))
        madx.input('replace, element=q%01d2, by=q%01d2_new;'%(np.floor(pos/2),np.floor(pos/2)))
    else:
        madx.input('q%01d1_new: quadrupole,L=quadrupoleLength, K1= 2.78/cellLength/quadrupoleLength;' %(np.floor(pos/2)))
        madx.input('replace, element=q%01d1, by=q%01d1_new;'%(np.floor(pos/2),np.floor(pos/2)))
    madx.input('endedit;')
    madx.input('makethin, sequence=FODO_ring;')
    madx.input('use, sequence=FODO_ring;')
#    madx.input('''
#        match, sequence=PS;
#        vary, name= kd, step= 0.00001;
#        vary, name= kf, step= 0.00001;
#        global,sequence=PS,Q1= 6.10;
#        global,sequence=PS,Q2= 6.10;
#        jacobian, calls = 50000, tolerance=1.0e-15;
#        endmatch;
#        ''' )
        
    madx.twiss()
    
    return madx, match_flag

def quick_match(madx,q1,q2):
    madx.input('''
        match, sequence=PS;
        vary, name= kd, step= 0.00001;
        vary, name= kf, step= 0.00001;
        global,sequence=PS,Q1= 6.0'''+q1+''';
        global,sequence=PS,Q2= 6.'''+q2+''';
        jacobian, calls = 50000, tolerance=1.0e-15;
        endmatch;
        ''' )   
    madx.twiss()

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
#    madx.input('''
#        match, sequence=PS;
#        vary, name= kd, step= 0.00001;
#        vary, name= kf, step= 0.00001;
#        global,sequence=PS,Q1= 6.10;
#        global,sequence=PS,Q2= 6.10;
#        jacobian, calls = 50000, tolerance=1.0e-15;
#        endmatch;
#        ''' )
      
    madx.twiss()
    if madx.table.summ.Q1 >= 6.15 :
        print('matching failed')
        match_flag = True
    
    return madx, match_flag

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

def find_best_conf(s,betx,bety,dx):
    linspace = np.linspace(4,628,2000)
    
    ndims = np.ndim(betx)
   
    
    if ndims == 2:
        BetaBeating = np.zeros((np.shape(betx)[0],))
        for i in range(betx.shape[0]):
            gotten_beta1 = make_interpolate_function(s,betx[i,:])
            gotten_beta2 = make_interpolate_function(s,bety[i,:])
            gotten_disp = make_interpolate_function(s,dx[i,:])
            
            BetaBeating[i] = np.mean([np.std(gotten_beta1(linspace)),np.std(gotten_beta2(linspace))])#,0.4*np.std(gotten_disp(linspace))])
            
    
    elif ndims == 3:
        BetaBeating = np.zeros((np.shape(betx)[0],np.shape(betx)[1],))
        for i in range(betx.shape[0]):
             for j in range(betx.shape[1]):
                gotten_beta1 = make_interpolate_function(s,betx[i,j,:])
                gotten_beta2 = make_interpolate_function(s,bety[i,j,:])
                gotten_disp = make_interpolate_function(s,dx[i,j,:])
                
                BetaBeating[i,j] = np.mean([np.std(gotten_beta1(linspace)),np.std(gotten_beta2(linspace))])#,0.4*np.std(gotten_disp(linspace))])

    else:
        pass
    
    idx = np.nonzero(BetaBeating == BetaBeating.min())
    
    if np.size(idx) >= 5:
        idx = np.array([0,0,])
        #print('nani', BetaBeating, BetaBeating.min(), BetaBeating == BetaBeating.min())
        
    if np.size(idx) >= 4:
        return idx[0], BetaBeating[idx][0]
    else:
        return idx, BetaBeating[idx][0]

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


def get_madx_mean1(madx,idx,s_operational,s_possible):
    madx,flag = add_wmarkers(madx,s_operational,s_possible[idx])
    print(tester_func(madx))
    std = get_mean_optics(madx)
    madx,flag = remove_wmarkers(madx,np.append(s_operational,s_possible[idx]),s_possible[idx]) 
    print(tester_func(madx))

    return std  

def get_madx_mean2(madx,idx,s_operational,s_possible):
    madx,flag = remove_wmarkers(madx,s_operational,s_operational[idx])
    print(tester_func(madx))
    std = get_mean_optics(madx)
    madx,flag = add_wmarkers(madx,np.delete(s_operational,idx),s_operational[idx]) 
    print(tester_func(madx))

    return std    

def get_madx_mean3(madx,idx,s_operational,s_possible):
    madx,flag = add_wmarkers(madx,s_operational,s_possible[idx[0]])
    madx,flag = remove_wmarkers(madx,s_operational,s_operational[idx[1]])
    std = get_mean_optics(madx)
    madx,flag = remove_wmarkers(madx,np.append(s_operational,s_possible[idx[0]]),s_possible[idx[0]])
    madx,flag = add_wmarkers(madx,np.delete(s_operational,idx[1]),s_operational[idx[1]]) 

    return std 

def get_madx_mean4(madx,idx,s_operational,s_possible):
    madx,flag = add_wmarkers(madx,s_operational,s_possible[idx[0]])
    madx,flag = add_wmarkers(madx,s_operational,s_possible[idx[1]])
    std = get_mean_optics(madx)
    madx,flag = remove_wmarkers(madx,np.append(s_operational,s_possible[idx[0]]),s_possible[idx[0]])
    madx,flag = remove_wmarkers(madx,np.append(s_operational,s_possible[idx[1]]),s_possible[idx[1]])

    return std 

def get_madx_mean5(madx,idx,s_operational,s_possible):
    madx,flag = remove_wmarkers(madx,s_operational,s_operational[idx[0]])
    madx,flag = remove_wmarkers(madx,s_operational,s_operational[idx[1]])
    std = get_mean_optics(madx)
    madx,flag = add_wmarkers(madx,np.delete(s_operational,idx[0]),s_operational[idx[0]]) 
    madx,flag = add_wmarkers(madx,np.delete(s_operational,idx[1]),s_operational[idx[1]]) 

    return std 

def tester_func(madx):
    temp_list = []
    for i,elem in enumerate(madx.table.twiss.name):
        if elem.startswith('pr.q'):
            temp_list.append(elem[6:8])
    return temp_list

def integrate_bend(posz, madx,Bx_err,angle,L,pos):
    
    #pos=0
    pos_l = []
    for j in range(5):
        pos_temp = (2*np.sin(angle/2)/L)*(Bx_err[posz[j]]/np.sqrt(madx.table.twiss.betx[posz[j]]))*np.cos(2*np.pi*np.abs(madx.table.twiss.mux[posz[j]] - madx.table.twiss.mux) - np.pi*6.1)
        pos_l.append(pos_temp)

    return (pos_l[0] + 2*pos_l[1] + 2*pos_l[2] + 2*pos_l[3] + pos_l[4])*L/8

def integrate_bend2(posz, madx,Bx_err,angle,L,pos):
    
    #pos=0
    pos_l = []
    for j in range(5):
        pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(madx.table.twiss.betx[posz[j]]))*np.sin(2*np.pi*np.abs(madx.table.twiss.mux[posz[j]] - madx.table.twiss.mux) - np.pi*6.1)
        pos_l.append(pos_temp)

    return (pos_l[0] + 2*pos_l[1] + 2*pos_l[2] + 2*pos_l[3] + pos_l[4])*L/8

def integrate_bend_FODO2(posz, madx,Bx_err,angle,L,pos):
    
    #pos=0
    pos_l = []
    for j in range(3):
        pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(madx.table.twiss.betx[posz[j]]))*np.sin(2*np.pi*np.abs(madx.table.twiss.mux[posz[j]] - madx.table.twiss.mux) - np.pi*madx.table.summ.q1)
        pos_l.append(pos_temp)

    return (pos_l[0] + 2*pos_l[1] + pos_l[2])*L/4


def integrate_bend_FODO(posz, madx,Bx_err,angle,L,pos):
    
    #pos=0
    pos_l = []
    for j in range(3):
        pos_temp = (2*np.sin(angle/2)/L)*(Bx_err[posz[j]]/np.sqrt(madx.table.twiss.betx[posz[j]]))*np.cos(2*np.pi*np.abs(madx.table.twiss.mux[posz[j]] + madx.table.twiss.mux[pos] - madx.table.twiss.mux) - np.pi*madx.table.summ.q1)
        pos_l.append(pos_temp)

    return (pos_l[0] + 2*pos_l[1] + pos_l[2])*L/4

def get_dispersion_beating_FODO(madx,Bx_err,pos):
   
    pos_bend = []
    j = 0
    
    temp_bend1=[]
    for i,elem in enumerate(madx.table.twiss.name):
        if j >=3:
            j=0
            pos_bend.append(temp_bend1)
            temp_bend1=[]     
            
        if elem.startswith('b') and elem.endswith(':1'):
            temp_bend1.append(i)
            j=j+1
        
    
    L_F = madx.eval('dipoleLength')
    angle_F = madx.eval('myAngle')

        
    Dx_pos_bend = np.sum(np.asarray([integrate_bend_FODO(i,madx,Bx_err,angle_F,L_F,pos)
                    for i in pos_bend]), axis = 0)
    
    
    Dx_first = (1/(4*np.sin(np.pi*madx.table.summ.q1))) * (Dx_pos_bend)
    
    #Dx_second = 0.5*madx.table.twiss.dx*Bx_err/madx.table.twiss.betx
               
    return Dx_first*madx.table.twiss.betx

def get_dispersion_beating_FODO2(madx,Bx_err,pos, dQ):
   
    pos_bend = []
    j = 0
    
    temp_bend1=[]
    for i,elem in enumerate(madx.table.twiss.name):
        if j >=3:
            j=0
            pos_bend.append(temp_bend1)
            temp_bend1=[]     
            
        if elem.startswith('b') and elem.endswith(':1'):
            temp_bend1.append(i)
            j=j+1
        
    
    L_F = madx.eval('dipoleLength')
    angle_F = madx.eval('myAngle')
        
    Dx_pos_bend = np.sum(np.asarray([integrate_bend_FODO2(i,madx,Bx_err,angle_F,L_F,pos)
                    for i in pos_bend]), axis = 0)
    
    
    Dx_first = (np.pi*dQ/(2*np.sin(np.pi*madx.table.summ.q1))) * (Dx_pos_bend)
    
    #Dx_second = 0.5*madx.table.twiss.dx*Bx_err/madx.table.twiss.betx
               
    return Dx_first*madx.table.twiss.betx

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

def get_dispersion_beating(int_steps,madx,Bx_err,pos):
   
    pos_bend, neg_bend = get_fdBend(int_steps,madx)
    
    L_F = madx.eval('L_F')
    L_D = madx.eval('L_D')
    angle_F = madx.eval('angle_F')
    angle_D = madx.eval('angle_D') 
        
    Dx_pos_bend = np.sum(np.asarray([integrate_bend(i,madx,Bx_err,angle_F,L_F,pos)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_bend(i,madx,Bx_err,angle_D,L_D,pos)
                    for i in neg_bend]), axis = 0)
    
    Dx_first = (1/(4*np.sin(np.pi*6.1))) * (Dx_pos_bend + Dx_neg_bend)
               
    return Dx_first*madx.table.twiss.betx

def get_dispersion_beating2(madx,Bx_err,pos, dQ):
   
    pos_bend = []
    neg_bend = []
    j = k = 0
    
    temp_bend1=[]
    temp_bend2=[]
    for i,elem in enumerate(madx.table.twiss.name):
        for l in range(1,6):
            if j >=5:
                j=0
                pos_bend.append(temp_bend1)
                temp_bend1=[]
                
            if k >=5:
                k=0
                neg_bend.append(temp_bend2)
                temp_bend2=[]
            
            if elem.startswith('pr.bh') and elem.endswith('f..'+str(l)+':1'):
                temp_bend1.append(i)
                j=j+1
        
            if elem.startswith('pr.bh') and elem.endswith('d..'+str(l)+':1'):
                temp_bend2.append(i)
                k=k+1
            
    
    
    
    L_F = madx.eval('L_F')
    L_D = madx.eval('L_D')
    angle_F = madx.eval('angle_F')
    angle_D = madx.eval('angle_D')
    
#    if (m % 2) == 0:
#        k1 = madx.eval('kf')  
#    elif (m % 2) == 1:
#        k1 = madx.eval('kd')  
        
    Dx_pos_bend = np.sum(np.asarray([integrate_bend2(i,madx,Bx_err,angle_F,L_F,pos)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_bend2(i,madx,Bx_err,angle_D,L_D,pos)
                    for i in neg_bend]), axis = 0)
    
    Dx_first = (np.pi*dQ/(2*np.sin(np.pi*6.1))) * (Dx_pos_bend + Dx_neg_bend)
    
    #Dx_second = 0.5*madx.table.twiss.dx*Bx_err/madx.table.twiss.betx
               
    return Dx_first*madx.table.twiss.betx


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


def integrate_bend20(phase_off,posz, madx,Bx_err,angle,L,pos,dQ):
    
    #pos=0
    pos_l = []
    for j in range(5):
        pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(madx.table.twiss.betx[posz[j]] + Bx_err[posz[j]]))*(np.cos(2*np.pi*np.abs( madx.table.twiss.mux[posz[j]] - madx.table.twiss.mux - phase_off) - np.pi*(madx.table.summ.q1 + dQ)))
        pos_l.append(pos_temp)

    return (pos_l[0] + 2*pos_l[1] + 2*pos_l[2] + 2*pos_l[3] + pos_l[4])*L/8

def get_dispersion_beating20(phase_off,madx,Bx_err,pos, dQ):
   
    pos_bend = []
    neg_bend = []
    j = k = 0
    
    temp_bend1=[]
    temp_bend2=[]
    for i,elem in enumerate(madx.table.twiss.name):
        for l in range(1,6):
            if j >=5:
                j=0
                pos_bend.append(temp_bend1)
                temp_bend1=[]
                
            if k >=5:
                k=0
                neg_bend.append(temp_bend2)
                temp_bend2=[]
            
            if elem.startswith('pr.bh') and elem.endswith('f..'+str(l)+':1'):
                temp_bend1.append(i)
                j=j+1
        
            if elem.startswith('pr.bh') and elem.endswith('d..'+str(l)+':1'):
                temp_bend2.append(i)
                k=k+1
            
    
    
    
    L_F = madx.eval('L_F')
    L_D = madx.eval('L_D')
    angle_F = madx.eval('angle_F')
    angle_D = madx.eval('angle_D')
            
    Dx_pos_bend = np.sum(np.asarray([integrate_bend20(phase_off,i,madx,Bx_err,angle_F,L_F,pos,dQ)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_bend20(phase_off,i,madx,Bx_err,angle_D,L_D,pos,dQ)
                    for i in neg_bend]), axis = 0)
    
    Dx_first = (Dx_pos_bend + Dx_neg_bend)
               
    return Dx_first


def check_optics_add_one(Position,FODO_flag,phase_off,dk_factor):  #Position is a number 1-100
    
    Position = Position - 1  #easier for python
    
    if Position not in range(100):
        print('Choose a number between 1-100')
        return 0
    
    ssz = get_positions_of_quads() 
    
    
    s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
    
    if FODO_flag:
        madx,posz = place_quads_wmarkers_FODO(13)
    else:
        madx,posz = place_quads_wmarkers(5,s_operational,range(100),ssz)
        
    pos = posz[Position]
   
    if FODO_flag:
        A=[-madx.eval('kf')*5/140,-madx.eval('kd')*5/140]
        delta_Q = [(-np.arccos(np.cos(2*np.pi*madx.table.summ.q1) - (A[Position%2]/2)*madx.table.twiss.betx[pos]*np.sin(2*np.pi*madx.table.summ.q1))/(2*np.pi)+np.round(madx.table.summ.q1)-madx.table.summ.q1),
            (np.arccos(np.cos(2*np.pi*madx.table.summ.q2) + (A[Position%2]/2)*madx.table.twiss.bety[pos]*np.sin(2*np.pi*madx.table.summ.q2))/(2*np.pi) +np.floor(madx.table.summ.q2)-madx.table.summ.q2)]

   
    else:
        A=[-madx.eval('kf')*dk_factor,-madx.eval('kd')*dk_factor]
        delta_Q = [(np.arccos(np.cos(2*np.pi*madx.table.summ.q1) - (A[Position%2]/2)*madx.table.twiss.betx[pos]*np.sin(2*np.pi*madx.table.summ.q1))/(2*np.pi)+np.floor(madx.table.summ.q1)-madx.table.summ.q1),
            (np.arccos(np.cos(2*np.pi*madx.table.summ.q2) + (A[Position%2]/2)*madx.table.twiss.bety[pos]*np.sin(2*np.pi*madx.table.summ.q2))/(2*np.pi) +np.floor(madx.table.summ.q2)-madx.table.summ.q2)]
    
    print(A[Position%2],madx.table.twiss.betx[pos], madx.table.summ.q1,madx.eval('kf')  )
    Bx_err = -A[Position%2]/(2*np.sin(2*np.pi*(madx.table.summ.q1 + delta_Q[0])))*madx.table.twiss.betx*madx.table.twiss.betx[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.mux[pos] - madx.table.twiss.mux) -2*np.pi*madx.table.summ.q1)               
    By_err = A[Position%2]/(2*np.sin(2*np.pi*(madx.table.summ.q2 + delta_Q[1])))*madx.table.twiss.bety*madx.table.twiss.bety[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.muy[pos] - madx.table.twiss.muy) -2*np.pi*madx.table.summ.q2) 
    
#    if FODO_flag:
#        Dx_err = get_dispersion_beating_FODO(madx,Bx_err,pos)
#        Dx_err2 = get_dispersion_beating_FODO2(madx,Bx_err,pos, delta_Q[0])
#    else:
#        Dx_err = get_dispersion_beating(madx,Bx_err,pos)
#        Dx_err2 = get_dispersion_beating2(madx,Bx_err,pos, delta_Q[0])
    
#    first_Dx_err = Dx_err/np.sqrt(madx.table.twiss.betx)
#    second_Dx_err = (0.5*madx.table.twiss.dx*Bx_err/madx.table.twiss.betx)
#    third_Dx_err = np.sqrt(madx.table.twiss.betx)*(-0.25*(np.sqrt(madx.table.twiss.betx)*(madx.table.twiss.dpx + madx.table.twiss.dx*(madx.table.twiss.alfx/madx.table.twiss.betx)) - (1/np.tan(np.pi*madx.table.summ.q1))*madx.table.twiss.dx ) * madx.table.twiss.betx[pos]*A[Position%2])
#    
        
#    first_Dx_err = Dx_err/np.sqrt(madx.table.twiss.betx)
#    second_Dx_err = Dx_err2/np.sqrt(madx.table.twiss.betx)
#    third_Dx_err = (Bx_err/(madx.table.twiss.betx*2))*(madx.table.twiss.dx-np.mean(madx.table.twiss.dx))
#    fourth_Dx_err = (-np.pi*delta_Q[0]*(np.sqrt(madx.table.twiss.betx)/np.tan(np.pi*(madx.table.summ.q1))))*(madx.table.twiss.dx-np.mean(madx.table.twiss.dx))
#    
#        
#    Full_Dx_err =  first_Dx_err + second_Dx_err + third_Dx_err + fourth_Dx_err
    
    temp_betx = madx.table.twiss.betx
    temp_bety = madx.table.twiss.bety
    temp_disp = madx.table.twiss.dx*madx.eval('beam->beta')
    temp_Q = madx.table.summ.q1
    temp_Qy = madx.table.summ.q2    
        
    first_part = (1/(2*np.sin(np.pi*(temp_Q + delta_Q[0]))))*(np.sqrt(temp_betx + Bx_err))      

    second_part =  get_dispersion_beating20(phase_off,madx,Bx_err,pos, delta_Q[0])
           
    Full_Dx_err =  first_part*second_part - temp_disp 
    
    #Full_Dx_err = np.roll(Full_Dx_err,pos)

    
    
    if FODO_flag:
        madx,flag = add_wmarkers_FODO(madx,Position)
    else:        
        #madx,flag = add_wmarkers(madx,s_operational,Position,dk_factor)
        madx,flag = remove_wmarkers(madx,s_operational,Position)
    
    
    temp_betx2 = madx.table.twiss.betx
    temp_bety2 = madx.table.twiss.bety
    temp_disp2 = madx.table.twiss.dx*madx.eval('beam->beta')
    temp_Q2 = madx.table.summ.q1
    temp_Q2y = madx.table.summ.q2
    
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
    
#    plt.figure(2)
#    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,first_part,'b')
#    plt.axvline(x= ssz[Position], linestyle=':', c='k')
#    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
#    plt.title('Optics check madx/ equations, section '+str(Position+1))
#    plt.xlabel('s (m)')
#    plt.ylabel(r'$\Delta D_x (m)$')
#    plt.legend(['madx','eqs'])
#    plt.savefig('img/dispersion_beating/dispx_'+str(Position+1)+'.png',dpi=300)
#    plt.show()
    
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
    
#    plt.figure(4)
#    plt.plot(madx.table.twiss.s,temp_disp2,'r',madx.table.twiss.s,first_part,'b')
#    plt.axvline(x= ssz[Position], linestyle=':', c='k')
#    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
#    plt.title('Optics check madx/ equations, section '+str(Position+1))
#    plt.xlabel('s (m)')
#    plt.ylabel(r'$\Delta D_x (m)$')
#    plt.legend(['madx','first term'])
#    plt.savefig('img/dispersion_beating/phase/first_dispx_'+str(Position+1)+'.png',dpi=300)
#    plt.show()
##    
#    plt.figure(5)
#    plt.plot(madx.table.twiss.s,temp_disp2,'r',madx.table.twiss.s,second_part,'b')
#    plt.axvline(x= ssz[Position], linestyle=':', c='k')
#    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
#    plt.title('Optics check madx/ equations, section '+str(Position+1))
#    plt.xlabel('s (m)')
#    plt.ylabel(r'$\Delta D_x (m)$')
#    plt.legend(['madx','second term'])
#    plt.savefig('img/dispersion_beating/phase/second_dispx_'+str(Position+1)+'.png',dpi=300)
#    plt.show()
#    
#    plt.figure(6)
#    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,third_Dx_err,'b')
#    plt.axvline(x= ssz[Position], linestyle=':', c='k')
#    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
#    plt.title('Optics check madx/ equations, section '+str(Position+1))
#    plt.xlabel('s (m)')
#    plt.ylabel(r'$\Delta D_x (m)$')
#    plt.legend(['madx','third term'])
#    plt.savefig('img/dispersion_beating/dispersion_terms/third_dispx_'+str(Position+1)+'.png',dpi=300)
#    plt.show()
#    
#    plt.figure(7)
#    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,fourth_Dx_err,'b')
#    plt.axvline(x= ssz[Position], linestyle=':', c='k')
#    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
#    plt.title('Optics check madx/ equations, section '+str(Position+1))
#    plt.xlabel('s (m)')
#    plt.ylabel(r'$\Delta D_x (m)$')
#    plt.legend(['madx','fourth term'])
#    plt.savefig('img/dispersion_beating/dispersion_terms/fourth_dispx_'+str(Position+1)+'.png',dpi=300)
#    plt.show()
    
    return Bx_err, By_err, Full_Dx_err, delta_Q[0]

def check_dispersion(Quad,int_steps,FODO_flag):
    
    ssz = get_positions_of_quads() 
    s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
    
    
    if FODO_flag:
        madx,posz = place_quads_wmarkers_FODO(int_steps)
        disp1 = calc_disp_FODO(int_steps,madx)
    else:
        madx,posz = place_quads_wmarkers(int_steps,s_operational,range(100),ssz)
        #madx,flag = add_wmarkers(madx,s_operational,Quad-1,1)
        disp1 = calc_disp(int_steps,madx)
    disp2 = madx.table.twiss.dx*madx.eval('beam->beta')
    
    dB = (disp1-disp2)
     
    plt.figure(8)
    plt.plot(madx.table.twiss.s,disp2,'b',madx.table.twiss.s,disp1,'r')
    plt.title('bare PS lattice, without absolute value')
    plt.xlabel('s (m)')
    plt.ylabel(r'$D_x (m)$')
    plt.legend(['madx','dispersion equation'])
    plt.savefig('img/dispersion_beating/abs_value_test2.png',dpi=300)
    plt.show()
    
    plt.figure(9)
    plt.plot(madx.table.twiss.s,(disp1-disp2),'r')
    #plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta D_x (m)$')
    plt.legend(['madx','calc disp'])
    #plt.savefig('img/dispersion_beating/dispersion_terms/fourth_dispx_'+str(Position+1)+'.png',dpi=300)
    plt.show()
    
    return disp1, np.mean(dB)

def get_phase_advance(mux, bx, betx, s):
    
    integral = np.zeros(len(mux))
    for i in range(1,len(s)):
        integral[i] = integral[i-1] + (s[i]-s[i-1])*((bx[i]/(betx[i]**2)) + (bx[i-1]/(betx[i-1]**2)))/2 #- (s[i]-s[i-1])*((2*(bx[i]**2)/(betx[i]**3)) + (2*(bx[i]**2)/(betx[i]**3)))/2
        
    phase_advance = mux - integral/(2*np.pi)
    
    return phase_advance
    
    
    
    