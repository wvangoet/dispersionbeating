import numpy as np
import pickle
#import pandas
from zoopt import ExpOpt
from scipy.interpolate import interp1d
from cpymad.madx import Madx
from cl2pd import madx as cl2madx
import matplotlib.pyplot as plt
from dispersion import *

def place_quads_wmarkers(pos,s_pos,ssz):
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
    madx.input('select, flag=makethin, CLASS=SBEND, THICK= false, SLICE =5;')
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


def add_wmarkers(madx,s_ope,pos):
    match_flag = False
    madx.input('seqedit, sequence = PS;')
    
    if pos in s_ope:
        if pos == 99: 
            madx.input('PR.QDN00_2: MULTIPOLE, KNL:={0,kd};')
            madx.input('replace, element=MARK100_2, by=PR.QDN00_2;')
        elif (pos % 2) == 1: 
            madx.input('PR.QDN%02d_2: MULTIPOLE, KNL:={0,kd};' %(pos+1))
            madx.input('replace, element=MARK%02d_2, by=PR.QDN%02d_2;' %(pos+1,pos+1))
        else:
            madx.input('PR.QFN%02d_2: MULTIPOLE, KNL:={0,kf};' %(pos+1))
            madx.input('replace, element=MARK%02d_2, by=PR.QFN%02d_2;' %(pos+1,pos+1))
    else:
        if pos == 99: 
            madx.input('PR.QDN00: MULTIPOLE, KNL:={0,kd};')
            madx.input('replace, element=MARK100, by=PR.QDN00;')
        elif (pos % 2) == 1: 
            madx.input('PR.QDN%02d: MULTIPOLE, KNL:={0,kd};' %(pos+1))
            madx.input('replace, element=MARK%02d, by=PR.QDN%02d;' %(pos+1,pos+1))
        else:
            madx.input('PR.QFN%02d: MULTIPOLE, KNL:={0,kf};' %(pos+1))
            madx.input('replace, element=MARK%02d, by=PR.QFN%02d;' %(pos+1,pos+1))
    madx.input('endedit;')
    madx.input('use, sequence=PS;')
        
    madx.twiss()
    if madx.table.summ.Q1 >= 6.15 :
        print('matching failed')
        match_flag = True
    
    return madx, match_flag



def integrate_bend(posz, madx,Bx_err,angle,L,pos,dQ):
    
    #pos=0
    pos_l = []
    for j in range(5):
        pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(madx.table.twiss.betx[posz[j]]) +
                    (Bx_err[posz[j]]/(2*np.sqrt(madx.table.twiss.betx[posz[j]]))))*(np.cos(2*np.pi*np.abs(madx.table.twiss.mux[posz[j]] - madx.table.twiss.mux) - np.pi*madx.table.summ.q1)+
                    np.pi*dQ*np.sin(2*np.pi*np.abs(madx.table.twiss.mux[posz[j]] - madx.table.twiss.mux) - np.pi*madx.table.summ.q1))
        pos_l.append(pos_temp)

    return (pos_l[0] + 2*pos_l[1] + 2*pos_l[2] + 2*pos_l[3] + pos_l[4])*L/8

def integrate_bend2(posz, madx,Bx_err,angle,L,pos,dQ):
    
    #pos=0
    pos_l = []
    for j in range(5):
        pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(madx.table.twiss.betx[posz[j]] + Bx_err[posz[j]]))*(np.cos(2*np.pi*np.abs(madx.table.twiss.mux[posz[j]] - madx.table.twiss.mux) - np.pi*(madx.table.summ.q1 + dQ)))
        pos_l.append(pos_temp)

    return (pos_l[0] + 2*pos_l[1] + 2*pos_l[2] + 2*pos_l[3] + pos_l[4])*L/8

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
            
    Dx_pos_bend = np.sum(np.asarray([integrate_bend2(i,madx,Bx_err,angle_F,L_F,pos,dQ)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_bend2(i,madx,Bx_err,angle_D,L_D,pos,dQ)
                    for i in neg_bend]), axis = 0)
    
    Dx_first = (Dx_pos_bend + Dx_neg_bend)
               
    return Dx_first

def get_dispersion_beating(madx,Bx_err,pos, dQ):
   
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
            
    Dx_pos_bend = np.sum(np.asarray([integrate_bend(i,madx,Bx_err,angle_F,L_F,pos,dQ)
                    for i in pos_bend]), axis = 0)
    
    Dx_neg_bend = np.sum(np.asarray([integrate_bend(i,madx,Bx_err,angle_D,L_D,pos,dQ)
                    for i in neg_bend]), axis = 0)
    
    Dx_first = (Dx_pos_bend + Dx_neg_bend)
               
    return Dx_first

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




def check_optics_add_one(Position,FODO_flag):  #Position is a number 1-100
    
    Position = Position - 1  #easier for python
    
    if Position not in range(100):
        print('Choose a number between 1-100')
        return 0
    
    ssz = get_positions_of_quads() 
    
    
    s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
    
    
    madx,posz = place_quads_wmarkers(s_operational,range(100),ssz)
        
    pos = posz[Position]
   
    
    A=[madx.eval('kf'),madx.eval('kd')]
    delta_Q = [(np.arccos(np.cos(2*np.pi*madx.table.summ.q1) - (A[Position%2]/2)*madx.table.twiss.betx[pos]*np.sin(2*np.pi*madx.table.summ.q1))/(2*np.pi)+np.floor(madx.table.summ.q1)-madx.table.summ.q1),
        (np.arccos(np.cos(2*np.pi*madx.table.summ.q2) + (A[Position%2]/2)*madx.table.twiss.bety[pos]*np.sin(2*np.pi*madx.table.summ.q2))/(2*np.pi) +np.floor(madx.table.summ.q2)-madx.table.summ.q2)]
    
    print(A[Position%2],madx.table.twiss.betx[pos], madx.table.summ.q1,madx.eval('kf')  )
    Bx_err = -A[Position%2]/(2*np.sin(2*np.pi*(madx.table.summ.q1 + delta_Q[0])))*madx.table.twiss.betx*madx.table.twiss.betx[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.mux[pos] - madx.table.twiss.mux) -2*np.pi*madx.table.summ.q1)               
    By_err = A[Position%2]/(2*np.sin(2*np.pi*(madx.table.summ.q2 + delta_Q[1])))*madx.table.twiss.bety*madx.table.twiss.bety[pos]*np.cos(4*np.pi*np.abs(madx.table.twiss.muy[pos] - madx.table.twiss.muy) -2*np.pi*madx.table.summ.q2) 
    
       
    temp_betx = madx.table.twiss.betx
    temp_bety = madx.table.twiss.bety
    temp_disp = madx.table.twiss.dx
    temp_Q = madx.table.summ.q1
    temp_Qy = madx.table.summ.q2
    
    #(1-np.pi*delta_Q[0]*(1/np.tan(np.pi*temp_Q)))
    
#    first_part = (1/(2*np.sin(np.pi*(temp_Q + delta_Q[0]))))*(np.sqrt(temp_betx)+(Bx_err/(2*np.sqrt(temp_betx))))
#    second_part =  get_dispersion_beating(madx,Bx_err,pos, delta_Q[0])
#           
#    Full_Dx_err =  first_part*second_part - temp_disp
#    
    
    first_part = (1/(2*np.sin(np.pi*(temp_Q + delta_Q[0]))))*(np.sqrt(temp_betx + Bx_err))
    second_part =  get_dispersion_beating2(madx,Bx_err,pos, delta_Q[0])
           
    Full_Dx_err =  first_part*second_part - temp_disp   
            
    madx,flag = add_wmarkers(madx,s_operational,Position)
    
    
    temp_betx2 = madx.table.twiss.betx
    temp_bety2 = madx.table.twiss.bety
    temp_disp2 = madx.table.twiss.dx
    temp_Q2 = madx.table.summ.q1
    temp_Q2y = madx.table.summ.q2
    
    print('madx tune diff:',temp_Q2-temp_Q,' calculated tune diff', delta_Q[0], ' dk = ',A[Position%2] )
    print('madx tune diff:',temp_Q2y-temp_Qy,' calculated tune diff', delta_Q[1], ' dk = ',A[Position%2] )
    
    #print(ssz,madx.table.twiss.s[posz])
    
    plt.figure(0)
    plt.plot(madx.table.twiss.s,temp_betx2-temp_betx,'r',madx.table.twiss.s,Bx_err,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_betx2[posz[Position]]-temp_betx[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta \beta_x (m)$')
    plt.legend(['madx','eqs'])
    plt.savefig('img/dispersion_beating/betx_'+str(Position+1)+'.png',dpi=300)
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
    plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta D_x (m)$')
    plt.legend(['madx','full eqs'])
    plt.savefig('img/dispersion_beating/full_dispx_'+str(Position+1)+'.png',dpi=300)
    plt.show()
    
    plt.figure(4)
    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,first_part,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta D_x (m)$')
    plt.legend(['madx','first term'])
    plt.savefig('img/dispersion_beating/dispersion_terms/first_dispx_'+str(Position+1)+'.png',dpi=300)
    plt.show()
    
    plt.figure(5)
    plt.plot(madx.table.twiss.s,temp_disp2-temp_disp,'r',madx.table.twiss.s,second_part,'b')
    plt.axvline(x= ssz[Position], linestyle=':', c='k')
    plt.scatter(madx.table.twiss.s[posz[Position]], temp_disp2[posz[Position]]-temp_disp[posz[Position]],s = 100,c='k')
    plt.title('Optics check madx/ equations, section '+str(Position+1))
    plt.xlabel('s (m)')
    plt.ylabel(r'$\Delta D_x (m)$')
    plt.legend(['madx','second term'])
    plt.savefig('img/dispersion_beating/dispersion_terms/second_dispx_'+str(Position+1)+'.png',dpi=300)
    plt.show()
    

    return Bx_err, By_err, Full_Dx_err


def integrate_bend_from_optics(posz,betx,Q1,mux,angle,L):

    pos_l = []
    for j in range(len(posz)):
        pos_temp = (2*np.sin(angle/2)/L)*(np.sqrt(np.abs(betx[posz[j]])))*np.cos(2*np.pi*np.abs(mux[posz[j]] - mux) - np.pi*Q1)
        pos_l.append(pos_temp)
        
    result = L/(3*(len(pos_l)-1)) *np.sum(np.array([ (pos_l[i-1]+4*pos_l[i]+pos_l[i+1]) for i in range(1,len(pos_l)-1,2)]), axis = 0)

    return result


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