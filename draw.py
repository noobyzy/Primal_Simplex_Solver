import numpy as np
import copy
import time
import re
from classicalSim import classicSimplex
from revisedSim import revSimplex
from auxsol import auxPro
from main import RawData_Read
import matplotlib.pyplot as plt

pathList = ['./data/test5','./data/test2', './data/test1']
methodList =['classical', 'revised']

TimeClassical = []
TimeRevised = []

label = []

iterationTime = 1000
for path in pathList:
    A, b, c, x_star = RawData_Read(path)

    label.append("dim = {}".format(str(max(A.shape))))
    for method in methodList:

        startTime = time.perf_counter()
        for _ in range(iterationTime):
            
            Aaux, B_inverse, baux, IterCountaux, baseVarList = auxPro(A,b)
            if method == 'classical':
                cla_result, cla_optimalVal, cla_IterCount = classicSimplex(Aaux, baux, c, baseVarList, 'Bland')
            else:
                rev_result, rev_optimalVal, rev_IterCount = revSimplex(Aaux, baux, c, B_inverse, baseVarList, 'Bland')
        endTime = time.perf_counter()
        if method == 'classical':
            TimeClassical.append(endTime-startTime)
        else:
            TimeRevised.append(endTime-startTime)

fig,ax = plt.subplots(figsize=(8,5),dpi=80)
width_1 = 0.4

ax.bar(np.arange(len(TimeClassical)),TimeClassical,width=width_1,tick_label=label,label = 'classical')

ax.bar(np.arange(len(TimeRevised))+width_1,TimeRevised,width=width_1,tick_label=label,label='revised')

ax.legend()

plt.title("Performance under different dimension")
plt.xlabel('dimension')
plt.ylabel('time\s')
plt.show()
