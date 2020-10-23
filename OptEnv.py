import get_match_and_beta
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from zoopt import Dimension, Parameter, Objective
from bayes_opt import BayesianOptimization
from cpymad.madx import Madx



class OptEnv(gym.Env):

    def __init__(self,ss, sz, focusing_list, solver, _n_iter, best_approx):
        self.dof = len(ss)
        self.rew = 2000
        self.x_prev = ss
        self.sz = sz
        self.focusing_list = focusing_list
        self.l = len(self.focusing_list)
        self.solver = solver
        self._n_iter = _n_iter
        self.t0 = time.time()
        self.timer = [0]
        self.values = [1000]
        # Spawn MAD-X process
        self.reset() 
        self.best_approx = best_approx
        
        if self.solver == 'ZOOpt':
            dim = self.dof
            #dim_bounds = self.Bounds_maker()
            dimobj = Dimension(dim, [[0,628.3185]]*dim, [True] * dim)
            self.parameter = Parameter(budget=self._n_iter,
                                       init_samples=[ss], exploration_rate = 0.25)
            self.step = Objective(self.step, dimobj)
        elif self.solver == 'BOBYQA':  # currently broken
            self.upper = np.multiply(np.ones((self.dof),),628.3185)
            self.lower = np.multiply(self.upper, 0)
        elif self.solver == 'Bayesian':  # currently unfinished
            dim = self.dof
            x = ['x1', 'x2' ,'x3' ,'x4' ,'x5' ,'x6' ,'x7' ,'x8' ,'x9' ,'x10' ,
                 'x11' ,'x12' ,'x13' ,'x14' ,'x15' ,'x16' ,'x17' ,'x18' ,'x19' ,'x20' ,
                 'x21' ,'x22' ,'x23' ,'x24' ,'x25' ,'x26' ,'x27' ,'x28' ,'x29' ,'x30' ,
                 'x31' ,'x32' ,'x33' ,'x34' ,'x35' ,'x36' ,'x37' ,'x38' ,'x39' , 'x40']#,
                 #'x41' ,'x42' ,'x43' ,'x44' ,'x45' ,'x46' ,'x47' ,'x48' ,'x49' , 'x50']
            bounds = {}
            for n in range(self.dof):
                bounds[x[n]] = (0 , 628.3185)
            self.pbounds = bounds
            self.optimizer = BayesianOptimization(
                    f=self.step2,
                    pbounds=self.pbounds,
                    random_state=2,
                    )
        else:
            self.bounds = [[0, 628.3185]] * self.dof
            
            

    def step(self, x):
        
        if self.solver == 'ZOOpt':
            x = x.get_x()
        else:
            pass
        
        self.madx.input('EXIT;') 
        self.reset()
          
        c1 = get_match_and_beta.getBeating(x, self.sz, self.madx, self.focusing_list, self.best_approx)
        a = (c1.match_and_beat())
        output = a[0]*100
        if output < self.rew:
            plt.figure(1)
            plt.clf() 
            self.x_best = a[5]
            self.rew = output
            plt.plot(a[3],a[4])
            print('new minimum found, saved to: img/'+self.solver+'.png')
            plt.tight_layout()
            plt.savefig('img/'+self.solver+'.png')
        self.x_prev = a[2]
        
        # If MAD-X has failed re-spawn process
        if a[1]:
            self.reset()

        self.timer.append(time.time() - self.t0)
        self.values.append(self.rew)

        # Objective function with a = [beam_size_x, beam_size_y, beam_size_z, loss, percentage, error_flag]
        
        #print('\n locations = ', self.x_prev,'\n')
        print('output =', output)
        # If objective function is best so far, update x_best with new best parameters
        
        
        return output

    def reset(self):
        """
         If MAD-X fails, re-spawn process
         """
        self.madx = Madx(stdout=False)
        self.madx.option(echo=False, warn=False, info=False, debug=False, verbose=False)
        self.madx.input('BEAM, PARTICLE=PROTON, PC = 2.14')
        self.madx.input('BRHO := BEAM->PC * 3.3356;')
        self.madx.call(file='ps_mu.seq')
        self.madx.call(file='ps_ss_mod.seq')
        self.madx.call(file='ps_50LeQ.str')
        self.madx.call(file='ps_pro_bare_machine.str')
        
        self.madx.call(file='remove_elements.seq')
        self.madx.input('seqedit, sequence = PS;')
        self.madx.input('select, flag=seqedit, class = MQNAAIAP;')
        self.madx.input('select, flag=seqedit, class = MQNABIAP;')
        self.madx.input('select, flag=seqedit, class = MQSAAIAP;')
        self.madx.input('select, flag=seqedit, class = QNSD;')
        self.madx.input('select, flag=seqedit, class = QNSF;')
        
        self.madx.input('use, sequence = PS;')
        self.madx.input('seqedit,sequence = PS;flatten;endedit;')
        self.madx.input('seqedit,sequence = PS;remove, element=SELECTED;endedit;')
        self.madx.input('endedit;')  
        self.madx.input('use, sequence = PS;')
        
    def Bounds_maker(self):
        size_section = 628.3185/self.dof
        bounds = [[(i-3)*size_section,i*size_section ] for i in range(3,41)]
        bounds = np.concatenate( ([[0,size_section*2]], bounds, [[size_section*38,size_section*40]]),0 )
        return bounds
    
       
    
    def step2(self, x1, x2 ,x3 ,x4 ,x5 ,x6 ,x7 ,x8 ,x9 ,x10 ,x11 ,x12 ,x13 ,x14 ,x15 ,x16 ,x17 ,x18 ,x19 ,x20,
              x21 ,x22 ,x23 ,x24 ,x25 ,x26 ,x27 ,x28 ,x29 ,x30 ,x31 ,x32 ,x33 ,x34 ,x35 ,x36 ,x37 ,x38 ,x39 ,
              x40):#, x41 ,x42 ,x43 ,x44 ,x45 ,x46 ,x47 ,x48 ,x49 , x50):
        
        x = [x1, x2 ,x3 ,x4 ,x5 ,x6 ,x7 ,x8 ,x9 ,x10 ,x11 ,x12 ,x13 ,x14 ,x15 ,x16 ,x17 ,x18 ,x19 ,x20,
              x21 ,x22 ,x23 ,x24 ,x25 ,x26 ,x27 ,x28 ,x29 ,x30 ,x31 ,x32 ,x33 ,x34 ,x35 ,x36 ,x37 ,x38 ,x39 ,
              x40]#, x41 ,x42 ,x43 ,x44 ,x45 ,x46 ,x47 ,x48 ,x49 , x50]
        
        self.madx.input('EXIT;') 
        self.reset()
        
        c1 = get_match_and_beta.getBeating(x, self.sz, self.madx, self.focusing_list, self.best_approx)
        a = (c1.match_and_beat())
        output = a[0]*100
        if output < self.rew:
            plt.clf() 
            self.x_best = a[2]
            self.rew = output
            plt.plot(a[3],a[4])
            print('new minimum found, saved to: img/'+self.solver+'.png')
            plt.tight_layout()
            plt.savefig('img/'+self.solver+'.png')
        self.x_prev = a[2]
        
        # If MAD-X has failed re-spawn process
        if a[1]:
            self.reset()
            
        self.timer.append(time.time() - self.t0)
        self.values.append(self.rew)

#        print('Iteration' + str(len(self.loss_all)))
#        print('Sigmax =' + str(a[0]) + ', Sigmay=' + str(a[1]))
#        print('Total =' + str(a[0] ** 2 + a[1] ** 2))
#        print('Percentage <5 um =' + str(a[4]))

        # Objective function with a = [beam_size_x, beam_size_y, beam_size_z, loss, percentage, error_flag]
        
        #print('\n locations = ', self.x_prev,'\n')
        print('output =', output)
        # If objective function is best so far, update x_best with new best parameters
        
        output = -1*output
        
        return output
