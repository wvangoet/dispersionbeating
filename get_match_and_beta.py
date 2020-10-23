"""
GET BEAM SIZE
R. Ramjiawan
Dec 2019
Optimise the beam size at injection based on seven quad strengths and two/four sextupole strengths
"""

import numpy as np
from BB_lib import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class getBeating:
    def __init__(self, s_now,  sz, madx, focusing_list, best_approx):
        self.num_quad = 10
        self.focusing_list = focusing_list
        self.s_now = s_now
        self.s_possible = [ 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 24, 25, 26,
                27, 30, 31, 34, 35, 38, 39, 44, 45, 48, 49, 50, 54, 55, 58, 59, 62,
                64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84, 85, 88, 89, 94, 95, 98,
                99]
        #self.s_possible = list(range(100))
        self.phase_advances_x = best_approx[0]
        self.phase_advances_y = best_approx[1]
        self.sz = np.asarray(sz)
        self.madx = madx
        #self.madx.input('twiss;')
        self.best_approx = best_approx
        self.linspace = np.linspace(4,628,2000)
        self.match_string = '''
        match, sequence=PS;
        vary, name= kd, step= 0.00001;
        vary, name= kf, step= 0.00001;
        global,sequence=PS,Q1= 6.10;
        global,sequence=PS,Q2= 6.10;
        jacobian, calls = 50000, tolerance=1.0e-15;
        endmatch;
        ''' 
        
        
    def match_and_beat(self):
        
        error_flag = 0
          
        
        try:
            self.madx.input('use, sequence=PS;')
            self.madx.input('seqedit, sequence = PS;')
            new_s_list = []
            disp_effects = []
            s_prev = np.zeros((4*self.num_quad,))
            for i in range(self.num_quad):
                s_idx = self.find_nearest(i,new_s_list)
                pos4,disp_effect = find_four_adept(s_idx,new_s_list,self.s_possible,self.phase_advances_x,self.phase_advances_y,0.15)
                disp_effects.append(disp_effect)
                
                j=0
                for s_idx in pos4:
                    if s_idx == 99: 
                        self.madx.input('PR.QDN00: MULTIPOLE, KNL:={0,kd};')
                        self.madx.input('install, element=PR.QDN00, at=' +str(623.834315)+';')
                    elif (s_idx % 2) == 1: 
                        self.madx.input('PR.QDN%02d: MULTIPOLE, KNL:={0,kd};' %(s_idx+1))
                        self.madx.input('install, element=PR.QDN%02d, at=' %(s_idx+1) +str(self.sz[s_idx])+';')
                    else:
                        self.madx.input('PR.QFN%02d: MULTIPOLE, KNL:={0,kf};' %(s_idx+1))
                        self.madx.input('install, element=PR.QFN%02d, at=' %(s_idx+1) +str(self.sz[s_idx])+';')
                        
                    s_prev[(4*i)+j] = self.sz[s_idx]
                    new_s_list.append(s_idx)
                    j=j+1
             
            #print(disp_effects)
            self.madx.input('endedit;')
            self.madx.input('use, sequence=PS;')
            self.madx.input(self.match_string)
            self.madx.twiss()
            self.madx.input('seqedit, sequence = PS;')
            for x in new_s_list:
                if x == 99: 
                    self.madx.input('remove, element=PR.QDN00;')
                elif (x % 2) == 1: 
                    self.madx.input('remove, element=PR.QDN%02d;' %(x+1))
                else:
                    self.madx.input('remove, element=PR.QFN%02d;' %(x+1))
            self.madx.input('endedit;')
        except RuntimeError:
            print('MAD-X Error occurred, re-spawning MAD-X process')
            error_flag = 1
            BetaBeating = 1000
            positions = [0,1,2,3]
            beta_function = [0,1,2,3]
               
        else:
            if self.madx.table.summ.Q1 >= 6.15 :
                print('match didnt work')
                print(self.madx.table.summ.Q1)
                print(new_s_list)
                BetaBeating = 10
                positions = [0,1,2,3]
                beta_function = [0,1,2,3]
            else:
                
                gotten_beta1 = self.make_interpolate_function(self.madx.table.twiss.S.T,self.madx.table.twiss.BETX)
                gotten_beta2 = self.make_interpolate_function(self.madx.table.twiss.S.T,self.madx.table.twiss.BETY)
                gotten_disp = self.make_interpolate_function(self.madx.table.twiss.S.T,self.madx.table.twiss.dx)
                
                # betabeating_list = []
                # for g in range(1,5):
                #     temp_beta = self.make_interpolate_function(self.best_approx[g][:,2],self.best_approx[g][:,3])
                #     temp = temp_beta(self.linspace) - gotten_beta(self.linspace)
                #     betabeating_list.append(np.sqrt(np.mean(np.multiply(temp,temp))))
                  
                # temp_beta1 = self.make_interpolate_function(self.best_approx[2][:,2],self.best_approx[2][:,3])
                # temp_beta2 = self.make_interpolate_function(self.best_approx[2][:,2],self.best_approx[2][:,3])
                # temp1 = temp_beta1(self.linspace) - gotten_beta1(self.linspace)
                # temp2 = temp_beta2(self.linspace) - gotten_beta2(self.linspace)
                
                BetaBeating = np.mean([np.std(gotten_beta1(self.linspace)),np.std(gotten_beta2(self.linspace)),0.8*np.std(gotten_disp(self.linspace))])
                
                positions = self.madx.table.twiss.S
                beta_function = self.madx.table.twiss.BETX
                
                print(new_s_list)
            
        return BetaBeating, error_flag, s_prev, positions, beta_function,new_s_list
    
    def find_nearest(self, i, s_new):
        
        array1 = self.sz

        value = self.s_now[i]
     
        idx1 = (np.abs(array1 - value)).argmin()
        if value != array1[idx1]:
            sign = int((array1[idx1] - value) / np.abs(value - array1[idx1]))
        else:
            sign = 1
        while True:
            if idx1 == 0:
                sign = 1
            elif idx1 == 99 :
                sign = -1
            
            
            if (idx1 in s_new) or not (idx1 in self.s_possible):   
                idx1 = idx1 + sign
            elif len(list(filter(lambda x: (x % 2 == 0), s_new ))) >= 20 and (idx1 % 2) == 0:
                idx1 = idx1 + sign
            elif len(list(filter(lambda x: (x % 2 == 1), s_new ))) >= 20 and (idx1 % 2) == 1:
                idx1 = idx1 + sign
            else:
                output = idx1
                break
        
        return output
    
    def delete_leq_rows(self, beta, names):
        temp_foc_list = np.concatenate(([self.focusing_list ,np.asarray(['FN99','DN00'])]))
        beta = beta[[i for i,e in enumerate(names) if e[:-2].upper() in self.best_approx[0][:,0]]]
        names = names[[i for i,e in enumerate(names) if e[:-2].upper() in self.best_approx[0][:,0]]]
        beta = beta[[i for i,e in enumerate(names) if e[:-2].upper() not in ['PR.Q'+str(leq) for leq in temp_foc_list]]]
    

        return beta
    
    
    def make_interpolate_function(self, pos, beta):
        
        uniQ_list = [0]
        for j in range(len(pos)-1):
            if pos[j] == pos[j+1]:
                pass
            else:
                uniQ_list.append(j+1)
                
        output = interp1d(list(pos[uniQ_list]),list(beta[uniQ_list]),kind='cubic')
    
        return output
    
    def get_actual_pos(self):
        new_s_list = []
        array1 = np.asarray(self.sz)
        for i in range(self.num_quad):
            s_pos = self.find_nearest(i,new_s_list)
            new_s_list.append(s_pos)
        output = np.sort([(np.abs(array1 - self.sz[value])).argmin() for value in new_s_list ])
        
        return output