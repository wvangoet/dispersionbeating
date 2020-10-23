import OptEnv as env_mod
import numpy as np
import pybobyqa
import pickle
#import pandas
from zoopt import ExpOpt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
#from bayes_opt import BayesianOptimization
from cl2pd import madx


solver_list = ['ZOOpt']

#solver_list = ['Nelder-Mead']#,'CG','BFGS','L-BFGS-B','TNC']#,'BOBYQA','ZOOpt','Bayesian']
timer = {}
values = {}
positions = {}
solutions = {}
#ss = np.asarray([20, 70, 88, 71, 84, 94, 95, 66, 98, 55, 54, 80, 59, 34, 21, 44, 4, 8, 5, 39, 30, 48, 38, 45, 49, 35, 85, 0, 10, 9, 31, 81, 50, 11, 14, 67, 77, 89, 27, 99])
# temp = np.asarray([8, 12, 0, 4, 44, 48, 20, 24, 39, 35, 27, 31, 5, 9, 17, 13, 89, 85, 77, 81, 80, 76, 84, 88, 45, 49, 25, 21, 59, 55, 95, 67, 38, 34, 26, 30, 66, 62, 58, 54])
# ss = temp[range(0,40,4)]
ss = np.sort(np.asarray(np.random.random_integers(0,99,10)))
#ss = list(range(50,100))
#ss = np.asarray([[i,i+1] for i in range(1,100,4)]).reshape((50,))

for solver in solver_list:
    n_iter = 100
    s=[]
    sz = []
    ssz = []
    
    focusing_list = np.asarray(['FN05','DW06','FN09','DN10','FW17','DW18','FN21','DW22',
                                'FW27','DW28','FW31','DW32','FN35','DN36','FN39','DN40','FN45','DN46',
                                'FN49','DN50','FN55','DW56','FW59','DW60','FN67','DN68','FN71',
                                'DN72','FN77','DN78','FN81','DN82','FN85','DN86','FN89','DN90','FN95','DN96',
                                'TN01','TN11','TN12','TN13','MN14','TN15','MN25','MN26','TN51','MN63','TN65','TN75'])
                                
                                #'FN99','DN00'])  
    
    focusing_list2 = np.asarray(['FN05','DW06','FN09','DN10','FW17','DW18','FN21','DW22',
                                'FW27','DW28','FW31','DW32','FN35','DN36','FN39','DN40','FN45','DN46',
                                'FN49','DN50','FN55','DW56','FW59','DW60','FN67','DN68','FN71',
                                'DN72','FN77','DN78','FN81','DN82','FN85','DN86','FN89','DN90','FN95','DN96',
                                'TN01','TN11','TN12','TN13','MN14','TN15','MN25','MN26','TN51','MN63','TN65','TN75','FN99','DN00'])
    
    # focusing_list = np.asarray(['FN05','DW06','FN09','DN10','FW17','DW18','FN21','DW22','FW27',
    #                             'DW28','FW31','DW32','FN35','DN36','FN39','DN40','FN45','DN46',
    #                             'FN49','DN50','FN55','DW56','FW59','DW60','FN67','DN68','FN71',
    #                             'DN72','FN77','DN78','FN81','DN82','FN85','DN86','FN89','DN90',
    #                             'FN95','DN96','FN99','DN00'])  
    
    f1 = madx.tfs2pd('pfw_25.txt')
    Table1=f1.iloc[0].TABLE
    table1 = Table1.to_numpy()
    
    
    # f2 = madx.tfs2pd('sol0.txt')
    # f3 = madx.tfs2pd('sol610.txt')
    # f4 = madx.tfs2pd('sol2.txt')
    # f5 = madx.tfs2pd('sol3.txt')
    
    # table2 = f2.iloc[0].TABLE.to_numpy()
    # table3 = f3.iloc[0].TABLE.to_numpy()
    # table4 = f4.iloc[0].TABLE.to_numpy()
    # table5 = f5.iloc[0].TABLE.to_numpy()
    
    # table2 = table2[[i for i,e in enumerate(table2[:,0]) if e in table1[:,0]],:]
    # table3 = table3[[i for i,e in enumerate(table3[:,0]) if e in table1[:,0]],:]
    # table4 = table4[[i for i,e in enumerate(table4[:,0]) if e in table1[:,0]],:]
    # table5 = table5[[i for i,e in enumerate(table5[:,0]) if e in table1[:,0]],:]
    
    # table2 = table2[[i for i,e in enumerate(table2[:,0]) if e not in ['PR.Q'+str(leq) for leq in focusing_list2]],:]
    # table3 = table3[[i for i,e in enumerate(table3[:,0]) if e not in ['PR.Q'+str(leq) for leq in focusing_list2]],:]
    # table4 = table4[[i for i,e in enumerate(table4[:,0]) if e not in ['PR.Q'+str(leq) for leq in focusing_list2]],:]
    # table5 = table5[[i for i,e in enumerate(table5[:,0]) if e not in ['PR.Q'+str(leq) for leq in focusing_list2]],:]
    
    idx_ssz = []
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
        
    # table2 = np.concatenate((table2[:122,:],table2[121,:].reshape((1,256)),table2[122:,:]))
    # table3 = np.concatenate((table3[:122,:],table3[121,:].reshape((1,256)),table3[122:,:]))
             
    # best_approx = np.asarray([table1,table2,table3,table4,table5])
    
    best_approx = [np.asarray(Table1['MUX'][idx_ssz])*2*np.pi,np.asarray(Table1['MUY'][idx_ssz])*2*np.pi]

    env = env_mod.OptEnv(ssz[ss], ssz, focusing_list, solver, n_iter, best_approx)
    
    if solver == 'BOBYQA':
        solution = pybobyqa.solve(env.step, ssz[ss], bounds=(env.lower, env.upper), seek_global_minimum=True)
    elif solver == 'ZOOpt':  
        solution = ExpOpt.min(env.step, env.parameter, plot=True)
    elif solver == 'Bayesian':
        env.optimizer.probe(params = ssz[ss], lazy = True)
        env.optimizer.maximize(n_iter = int(env._n_iter*0.7), init_points = int(env._n_iter*0.3) )
        solution = env.optimizer.max
        
    else:
        solution = minimize(env.step, ssz[ss], method=solver, bounds=env.bounds, options={'maxiter': n_iter,
                                                                                    'xtol': 2,
                                                                                    'adaptive': True
                                                                                        })

    
    timer[solver] = env.timer
    values[solver] = env.values
    positions[solver] = env.x_best
    solutions[solver] = solution

with open('variables60.pkl', 'wb') as f:
    pickle.dump([timer,values,positions,solutions], f)

    

[85, 81, 73, 77, 69, 65, 57, 61, 83, 79, 71, 75, 86, 82, 74, 78, 14, 10, 2, 6, 23, 19, 11, 15, 66, 62, 54, 58, 18, 22, 26, 30, 76, 72, 64, 68, 56, 52, 44, 48, 5, 1, 9, 13, 63, 59, 51, 55]