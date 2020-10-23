from cpymad.madx import Madx
import numpy as np
import matplotlib.pyplot as plt
from cl2pd import madx as cl2madx
import pandas as pd

match_string = '''
        match, sequence=PS;
        vary, name= kd, step= 0.00001;
        vary, name= kf, step= 0.00001;
        global,sequence=PS,Q1= 6.20;
        global,sequence=PS,Q2= 6.20;
        jacobian, calls = 100, tolerance=1.0e-21;
        endmatch;
        '''   

############## The cell where we first remove all LeQ's and then install the correct ones ##################

## Madx definition ----------------------------------------------------------------------------------------------------------------------
madx = Madx()
#madx = Madx()
madx.input('BEAM, PARTICLE=PROTON, PC = 2.14')
madx.input('BRHO := BEAM->PC * 3.3356;')

# using the latest model directly from EOS (if you are on SWAN)
madx.call(file='/eos/project-a/acc-models/public/ps/2021/ps_mu.seq')
madx.call(file='/eos/project-a/acc-models/public/ps/2021/ps_ss.seq')
madx.call(file='/eos/project-a/acc-models/public/ps/2021/scenarios/bare_machine/0_proton_injection_energy/ps_pro_bare_machine.str')
madx.call(file='/eos/project-a/acc-models/public/ps/supplementary/space_charge_simulations/remove_elements.seq')        

# removing the LEQ in addition to all other elements mentioned in the file remove_elements.seq
madx.input('select, flag=seqedit, class = MQNAAIAP;')
madx.input('select, flag=seqedit, class = MQNABIAP;')
madx.input('select, flag=seqedit, class = MQSAAIAP;')
madx.input('select, flag=seqedit, class = QNSD;')
madx.input('select, flag=seqedit, class = QNSF;')

madx.input('use, sequence = PS;')
madx.input('seqedit,sequence = PS;flatten;endedit;')
madx.input('seqedit,sequence = PS;remove, element=SELECTED;endedit;')

madx.input('use, sequence = PS;')
madx.input('twiss')

# DataFrame to calculate possible positions for LEQ in all SSs 
quad_positions = pd.DataFrame(index = [int(elem[2:4]) for elem in madx.table.twiss.NAME if elem.startswith('ss') and elem.endswith('start:1')], columns = ['twiss_index', 's_SS', 's_LEQ'])

# offsets to be added to the start of the SS to obtain location at which LEQ can be installed
# first value for short SS, second for long ones
delta = [1.486343, 2.886343]

# filling the DataFrame, using as index the number of the SS, s_SS is the s location at the start of each SS and s_LEQ the possoible s location for each quad 
for i,elem in enumerate(madx.table.twiss.NAME):
    if elem.startswith('ss') and elem.endswith('start:1'):
        ss = int(elem[2:4])
        quad_positions['twiss_index'].loc[ss] = i
        quad_positions['s_SS'].loc[ss] = madx.table.twiss.S[i]
        if (elem[3] == '1') or (elem[3] == '6'):
            quad_positions['s_LEQ'].loc[ss] = quad_positions['s_SS'].loc[ss] + delta[1]
        else:
            quad_positions['s_LEQ'].loc[ss] = quad_positions['s_SS'].loc[ss] + delta[0]
            
g = 2
ss = np.asarray([[i,i+1] for i in range(g,100,4)]).reshape((50,))   #---- selecting sections [2,3,6,7,10,11, ... ]
ss = np.where(ss==100,0,ss)

## Installing all LeQ's ----------------------------------------------------------------------------------------------------------------------
madx.input('use, sequence=PS;')
madx.input('seqedit, sequence = PS;')

for x in ss:
    if (x%2) == 0:
        madx.input('PR.QD%02d: MULTIPOLE, KNL:={0,kd}' %x)
        madx.input('install, element= PR.QD%02d, at=' %x + str(quad_positions['s_LEQ'].loc[x]))  ## defocusing LeQ if position is even
    elif (x%2) == 1:
        madx.input('PR.QF%02d: MULTIPOLE, KNL:={0,kf}' %x)
        madx.input('install, element= PR.QF%02d, at=' %x + str(quad_positions['s_LEQ'].loc[x]))  ## focusing LeQ if position is uneven
    else:
        pass
madx.input('endedit;')

#rematching
madx.input('use, sequence=PS;')
madx.input(match_string)
madx.input('twiss')