import numpy as np
import pickle
import matplotlib.pyplot as plt
from cl2pd import madx


plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['figure.figsize'] = [8.0, 6.0]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 300

plt.rcParams.update({'font.size': 20})

ss = np.asarray([ 1,  6,  6, 12, 13, 13, 13, 14, 23, 28, 29, 30, 31, 33, 34, 36, 37,
       43, 44, 44, 46, 49, 53, 53, 54, 54, 55, 57, 57, 61, 63, 63, 63, 64,
       70, 72, 72, 73, 73, 73, 73, 74, 76, 81, 81, 82, 85, 90, 91, 95])
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

f1 = madx.tfs2pd('twiss_no_error.txt')
Table1=f1.iloc[0].TABLE
table1 = Table1.to_numpy()

f2 = madx.tfs2pd('sol0.txt')
f3 = madx.tfs2pd('sol1.txt')
f4 = madx.tfs2pd('sol2.txt')
f5 = madx.tfs2pd('sol3.txt')

table2 = f2.iloc[0].TABLE.to_numpy()
table3 = f3.iloc[0].TABLE.to_numpy()
table4 = f4.iloc[0].TABLE.to_numpy()
table5 = f5.iloc[0].TABLE.to_numpy()

table2 = table2[[i for i,e in enumerate(table2[:,0]) if e in table1[:,0]],:]
table3 = table3[[i for i,e in enumerate(table3[:,0]) if e in table1[:,0]],:]
table4 = table4[[i for i,e in enumerate(table4[:,0]) if e in table1[:,0]],:]
table5 = table5[[i for i,e in enumerate(table5[:,0]) if e in table1[:,0]],:]

table2 = table2[[i for i,e in enumerate(table2[:,0]) if e not in ['PR.Q'+str(leq) for leq in focusing_list2]],:]
table3 = table3[[i for i,e in enumerate(table3[:,0]) if e not in ['PR.Q'+str(leq) for leq in focusing_list2]],:]
table4 = table4[[i for i,e in enumerate(table4[:,0]) if e not in ['PR.Q'+str(leq) for leq in focusing_list2]],:]
table5 = table5[[i for i,e in enumerate(table5[:,0]) if e not in ['PR.Q'+str(leq) for leq in focusing_list2]],:]
    
for i in range(10):
    j,_ = np.nonzero(table1 == 'PS0'+str(i)+'$END')
    ssz.append(float(table1[j,2] - 0.125))
    
for i in range(10,100):
    j,_ = np.nonzero(table1 == 'PS'+str(i)+'$END')
    ssz.append(float(table1[j,2] - 0.125))
ssz = np.asarray(np.sort(ssz))
    
table2 = np.concatenate((table2[:122,:],table2[121,:].reshape((1,256)),table2[122:,:]))
table3 = np.concatenate((table3[:122,:],table3[121,:].reshape((1,256)),table3[122:,:]))
         
best_approx = np.asarray([table1,table2,table3,table4,table5])

#%%

timer, values, positions, solutions = pickle.load(open('variables50.pkl','rb'))

for solver in timer:
    plt.plot(timer[solver],values[solver],label = solver)
    plt.xlabel('Time (s)')
    plt.ylabel('output (B2-B1)')
    plt.legend()
    plt.tight_layout()
    
plt.title('compare opt methods  (1000 iterations)')
#plt.xlim([0,400])  
plt.ylim([0,100])  
plt.show()


#%%
from BB_lib import *
from scipy.interpolate import interp1d
from cpymad.madx import Madx
import matplotlib.pyplot as plt
from cl2pd import madx as cl2madx
linspace = np.linspace(4,628,2000)


#temp2 = np.asarray([[i,i+1] for i in range(0,96,4)]).reshape((48,))
temp2 = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
temp = positions['ZOOpt']
#temp = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,95,96,99,100])-1

#temp = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1
#temp = range(40)

s_possible = np.asarray([ 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 24, 25, 26,
                27, 30, 31, 34, 35, 38, 39, 44, 45, 48, 49, 50, 54, 55, 58, 59, 62,
                64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84, 85, 88, 89, 94, 95, 98,
                99])

madx1,_ = place_quads_wmarkers(9,temp,s_possible,ssz)
# madx1,flag = add_wmarkers(madx1,temp,25)
# madx1,flag = add_wmarkers(madx1,temp,95)
pos = []
madx1.input('seqedit, sequence = PS;')
for s_idx in pos:
        if s_idx == 99: 
            madx1.input('PR.QDN100: MULTIPOLE, KNL:={0,kd};')
            madx1.input('install, element=PR.QDN100, at=' +str(623.834315-0.1)+';')
        elif (s_idx % 2) == 1: 
            madx1.input('PR.QDN1%02d: MULTIPOLE, KNL:={0,kd};' %(s_idx+1))
            madx1.input('install, element=PR.QDN1%02d, at=' %(s_idx+1) +str(ssz[s_idx]+0.01)+';')
        else:
            madx1.input('PR.QFN1%02d: MULTIPOLE, KNL:={0,kf};' %(s_idx+1))
            madx1.input('install, element=PR.QFN1%02d, at=' %(s_idx+1) +str(ssz[s_idx]+0.01)+';')
madx1.input('endedit;')
madx1.input('use, sequence=PS;')
madx1.input('''
    match, sequence=PS;
    vary, name= kd, step= 0.00001;
    vary, name= kf, step= 0.00001;
    global,sequence=PS,Q1= 6.10;
    global,sequence=PS,Q2= 6.10;
    jacobian, calls = 50000, tolerance=1.0e-15;
    endmatch;
    ''' )
madx1.twiss()
madx2,_ = place_quads_wmarkers(9,temp2,s_possible,ssz)


uniQ_list = [0]
pos1 = madx1.table.twiss.s
betay1 = madx1.table.twiss.bety
betax1 = madx1.table.twiss.betx
Dx1 = madx1.table.twiss.dx

for j in range(len(pos1)-1):
    if pos1[j] == pos1[j+1]:
        pass
    else:
        uniQ_list.append(j+1)


betax_int = interp1d(list(pos1[uniQ_list]),list(betax1[uniQ_list]),kind='cubic')
betay_int = interp1d(list(pos1[uniQ_list]),list(betay1[uniQ_list]),kind='cubic')
Dx_int = interp1d(list(pos1[uniQ_list]),list(Dx1[uniQ_list]),kind='cubic')
Betax1_cont = betax_int(linspace)
Betay1_cont = betay_int(linspace)
Dx1_cont = Dx_int(linspace)

print(get_mean_optics(madx1))


uniQ_list = [0]
pos2 = madx2.table.twiss.s
betay2 = madx2.table.twiss.bety
betax2 = madx2.table.twiss.betx
Dx2 = madx2.table.twiss.dx

for j in range(len(pos2)-1):
    if pos2[j] == pos2[j+1]:
        pass
    else:
        uniQ_list.append(j+1)


betax_int = interp1d(list(pos2[uniQ_list]),list(betax2[uniQ_list]),kind='cubic')
betay_int = interp1d(list(pos2[uniQ_list]),list(betay2[uniQ_list]),kind='cubic')
Dx_int = interp1d(list(pos2[uniQ_list]),list(Dx2[uniQ_list]),kind='cubic')
Betax2_cont = betax_int(linspace)
Betay2_cont = betay_int(linspace)
Dx2_cont = Dx_int(linspace)
print(get_mean_optics(madx2))

optics1 = [Betax1_cont,Betay1_cont,Dx1_cont]
optics2 = [Betax2_cont,Betay2_cont,Dx2_cont]
std1 = [np.std(Betax1_cont),np.std(Betay1_cont),np.std(Dx1_cont)]
std2 = [np.std(Betax2_cont),np.std(Betay2_cont),np.std(Dx2_cont)]

# A=madx1.eval('kd')/(2*np.sin(2*np.pi*6.1))
# pos_70 = [i for i,elem in enumerate(madx1.table.twiss.name) if elem.startswith('ps40$start') ]
# # pos_71 = [i for i,elem in enumerate(madx1.table.twiss.name) if elem.startswith('ps71$start') ]
# Bx_err_70 = -A*betax1*betax1[pos_70]*np.cos(4*np.pi*np.abs(madx1.table.twiss.mux[pos_70] - madx1.table.twiss.mux) -2*np.pi*6.1)
# # Bx_err_71 = -A*betax1*betax1[pos_71]*np.cos(4*np.pi*np.abs(madx1.table.twiss.mux[pos_71] - madx1.table.twiss.mux) -2*np.pi*6.1)
# By_err_70 = A*betay1*betay1[pos_70]*np.cos(4*np.pi*np.abs(madx1.table.twiss.muy[pos_70] - madx1.table.twiss.muy) -2*np.pi*6.1)
# # By_err_71 = A*betay1*betay1[pos_71]*np.cos(4*np.pi*np.abs(madx1.table.twiss.muy[pos_71] - madx1.table.twiss.muy) -2*np.pi*6.1)
# Dx_err_70 = -madx1.eval('kd')/(2*np.sin(np.pi*6.1))*Dx1*betax1[pos_70]*np.cos(2*np.pi*np.abs(madx1.table.twiss.mux[pos_70] - madx1.table.twiss.mux) -2*np.pi*6.1)
# # Dx_err_71 = -6.5*A*Dx1*Dx1[pos_71]*np.cos(2*np.pi*np.abs(madx1.table.twiss.mux[pos_71] - madx1.table.twiss.mux) -np.pi*6.1)

# pos_70 = [i for i,elem in enumerate(madx2.table.twiss.name) if elem.startswith('ps70$start') ]
# pos_71 = [i for i,elem in enumerate(madx2.table.twiss.name) if elem.startswith('ps71$start') ]
# Bx_err_70 = -A*betax2*betax2[pos_70]*np.cos(4*np.pi*np.abs(madx2.table.twiss.mux[pos_70] - madx2.table.twiss.mux) -2*np.pi*6.1)
# Bx_err_71 = -A*betax2*betax2[pos_71]*np.cos(4*np.pi*np.abs(madx2.table.twiss.mux[pos_71] - madx2.table.twiss.mux) -2*np.pi*6.1)
# By_err_70 = -A*betay2*betay2[pos_70]*np.sin(4*np.pi*np.abs(madx2.table.twiss.muy[pos_70] - madx2.table.twiss.muy) -2*np.pi*6.1)
# By_err_71 = -A*betay2*betay2[pos_71]*np.sin(4*np.pi*np.abs(madx2.table.twiss.muy[pos_71] - madx2.table.twiss.muy) -2*np.pi*6.1)
# Dx_err_70 = -5.5*A*Dx2*Dx2[pos_70]*np.cos(2*np.pi*np.abs(madx2.table.twiss.mux[pos_70] - madx2.table.twiss.mux) -np.pi*6.1)
# Dx_err_71 = -5.5*A*Dx2*Dx2[pos_71]*np.cos(2*np.pi*np.abs(madx2.table.twiss.mux[pos_71] - madx2.table.twiss.mux) -np.pi*6.1)



# optics3=[Bx_err_70,By_err_70,Dx_err_70]
texts = ['Beta_x', 'Beta_y', 'Dx']
for i in range(3):
    plt.figure(i)
    plt.plot(linspace,optics2[i],'b')
    #plt.plot(pos1,optics3[i],'r')
    plt.plot(linspace,optics1[i],'r')
    # plt.plot(ssz,[8]*len(ssz),'k')
    # plt.plot(ssz[temp],[8]*len(ssz[temp]),'bo')
    # plt.plot(ssz,[7]*len(ssz),'k')
    # plt.plot(ssz[temp2],[7]*len(ssz[temp2]),'ro')
    
    plt.legend(('PS std: %1.3f' %std2[i],'NEW std: %1.3f' %std1[i]),loc=1,frameon=True)
    #plt.legend(('NEW','PS'),loc=1,frameon=True)
    plt.ylabel(texts[i]+' (m)')
    plt.xlabel(r'Arc length $s$ (m)')
    plt.tight_layout()
    plt.show()
    #plt.savefig('img/pos70_'+texts[i]+'.png')
    plt.clf()
    
#%%
plt.figure(dpi = 100)
plt.plot(ssz[temp],[2]*len(ssz[temp]),'bo')
plt.plot(ssz[temp2],[1]*len(ssz[temp2]),'ro') 
plt.plot(ssz,[2]*len(ssz),'k')
plt.plot(ssz,[1]*len(ssz),'k')
  
plt.legend(('PS','NEW'),loc=1,frameon=True)
#plt.xlabel(r'Arc length $s$ (m)')
plt.ylim([0,30])
frame1 = plt.gca()
frame1.axes.yaxis.set_visible(False)

plt.tight_layout()
#plt.show()
plt.savefig('img/pos70_quads.png')

#%%
from BB_lib import *
import timeit
linspace = np.linspace(4,628,2000)

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


s_possible = np.asarray([ 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 24, 25, 26,
                27, 30, 31, 34, 35, 38, 39, 44, 45, 48, 49, 50, 54, 55, 58, 59, 62,
                64, 66, 67, 70, 71, 74, 76, 77, 80, 81, 84, 85, 88, 89, 94, 95, 98,
                99])


s_operational = np.asarray([5,6,9,10,17,18,21,22,27,28,31,32,35,36,39,40,45,46,49,50,55,56,59,60,67,68,71,72,77,78,81,82,85,86,89,90,95,96,99,100])-1

madx,posz = place_quads_wmarkers(s_operational,s_possible,ssz)

#---------------------------------------------------
#madx1,flag = remove_wmarkers(madx,s_operational,94)
#madx1,flag = add_wmarkers(madx1,s_operational,12)
#s_operational = np.delete(s_operational, 94)
madx1 = madx
#---------------------------------------------------

uniQ_list = [0]
pos1 = madx1.table.twiss.s
betay1 = madx1.table.twiss.bety
betax1 = madx1.table.twiss.betx
Dx1 = madx1.table.twiss.dx

start1 = timeit.timeit()

A=-madx.eval('kd')/(2*np.sin(2*np.pi*6.1))
pos_70 = [i for i,elem in enumerate(madx.table.twiss.name) if elem.startswith('pr.qdn90') ]
Bx_err_70 = -A*betax1*betax1[pos_70]*np.cos(4*np.pi*np.abs(madx1.table.twiss.mux[pos_70] - madx1.table.twiss.mux) -2*np.pi*6.1)
By_err_70 = A*betay1*betay1[pos_70]*np.cos(4*np.pi*np.abs(madx1.table.twiss.muy[pos_70] - madx1.table.twiss.muy) -2*np.pi*6.1)
Dx_err_70 = madx1.eval('kd')/(2*np.sin(np.pi*6.1))*Dx1*betax1[pos_70]*np.cos(2*np.pi*np.abs(madx1.table.twiss.mux[pos_70] - madx1.table.twiss.mux) -2*np.pi*6.1)
end1 = timeit.timeit()
print('formula time:', end1-start1)

for j in range(len(pos1)-1):
    if pos1[j] == pos1[j+1]:
        pass
    else:
        uniQ_list.append(j+1)



betax_int = interp1d(list(pos1[uniQ_list]),list(betax1[uniQ_list]),kind='cubic')
betay_int = interp1d(list(pos1[uniQ_list]),list(betay1[uniQ_list]),kind='cubic')
Dx_int = interp1d(list(pos1[uniQ_list]),list(Dx1[uniQ_list]),kind='cubic')
Betax1_cont = betax_int(linspace)
Betay1_cont = betay_int(linspace)
Dx1_cont = Dx_int(linspace)

print(get_mean_optics(madx1))

start2 = timeit.timeit()

#---------------------------------------------------
madx2,flag = remove_wmarkers(madx1,s_operational,89)
# s_operational = np.delete(s_operational, 95)
#madx2,flag = add_wmarkers(madx1,s_operational,12)
#s_operational = np.append(s_operational,12)
#---------------------------------------------------

end2 = timeit.timeit()

uniQ_list = [0]
pos2 = madx2.table.twiss.s
betay2 = madx2.table.twiss.bety
betax2 = madx2.table.twiss.betx
Dx2 = madx2.table.twiss.dx

for j in range(len(pos2)-1):
    if pos2[j] == pos2[j+1]:
        pass
    else:
        uniQ_list.append(j+1)


betax_int = interp1d(list(pos2[uniQ_list]),list(betax2[uniQ_list]),kind='cubic')
betay_int = interp1d(list(pos2[uniQ_list]),list(betay2[uniQ_list]),kind='cubic')
Dx_int = interp1d(list(pos2[uniQ_list]),list(Dx2[uniQ_list]),kind='cubic')
Betax2_cont = betax_int(linspace)
Betay2_cont = betay_int(linspace)
Dx2_cont = Dx_int(linspace)
print(get_mean_optics(madx2))



print('madx time:', end2-start2)

optics1 = [Betax1_cont,Betay1_cont,Dx1_cont]
optics2 = [Betax2_cont,Betay2_cont,Dx2_cont]


                        
optics3=[Bx_err_70,By_err_70,Dx_err_70]
texts = ['Beta_x', 'Beta_y', 'Dx']
for i in range(3):
    plt.figure(i)
    plt.plot(linspace,optics2[i]-optics1[i],'b')
    plt.plot(pos1,optics3[i],'r')
    #plt.plot(linspace,optics2[i],'r')
    # plt.plot(ssz,[8]*len(ssz),'k')
    # plt.plot(ssz[temp],[8]*len(ssz[temp]),'bo')
    # plt.plot(ssz,[7]*len(ssz),'k')
    # plt.plot(ssz[temp2],[7]*len(ssz[temp2]),'ro')
    
    plt.legend((r'$\Delta (PS-NEW)$','formula'),loc=1,frameon=True)
    #plt.legend(('NEW','PS'),loc=1,frameon=True)
    plt.ylabel(texts[i]+' (m)')
    plt.xlabel(r'Arc length $s$ (m)')
    plt.tight_layout()
    plt.show()
    #plt.savefig('img/new_approach/r_34_a_95_'+texts[i]+'.png',dpi=300)
    plt.clf()
    
for i in range(3):
    plt.figure(i)
    plt.plot(linspace,optics1[i],'b')
    plt.plot(linspace,optics2[i],'r')
    #plt.plot(linspace,optics2[i],'r')
    # plt.plot(ssz,[8]*len(ssz),'k')
    # plt.plot(ssz[temp],[8]*len(ssz[temp]),'bo')
    # plt.plot(ssz,[7]*len(ssz),'k')
    # plt.plot(ssz[temp2],[7]*len(ssz[temp2]),'ro')
    
    plt.legend(('OLD','NEW'),loc=1,frameon=True)
    #plt.legend(('NEW','PS'),loc=1,frameon=True)
    plt.ylabel(texts[i]+' (m)')
    plt.xlabel(r'Arc length $s$ (m)')
    plt.tight_layout()
    #plt.show()
    plt.savefig('img/new_approach/r_90_full_'+texts[i]+'.png',dpi=300)
    plt.clf()
    



