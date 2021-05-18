"""
This is the final project of SI152 2020Fall by Yuzy, id: 2018533124

Here, I implemented a Classical Simplex method
"""

import numpy as np
import copy
import time
import re
from classicalSim import classicSimplex
from revisedSim import revSimplex
from auxsol import auxPro

'''
read A, b, c and x_star from the given path
'''
def RawData_Read(pathstr):
    A = np.loadtxt(pathstr + "/A.csv",delimiter=',')
    b = np.loadtxt(pathstr + "/b.csv",delimiter=',')
    c = np.loadtxt(pathstr + "/c.csv",delimiter=',')
    x_star = np.loadtxt(pathstr + "/x_star.csv",delimiter=',')
    return A,b,c,x_star

'''
write the optimal solution to x_{}.csv;
write optimal objective value, number of iterations and total running time to classical_{}.txt
'''
def WriteResult(optSol, optVal, numIter, totalTime, met='classical'):
    if met == 'classical':
        np.savetxt('result/x_classical.csv', optSol, fmt='%.10f', delimiter = ',')
        f = open('result/classical_result.txt','w')
        lines = ['optimal objective value: {}\n'.format(optVal), 'number of pivots: {}\n'.format(numIter), 'total running time: {} s'.format('%.8f'%totalTime)]
        f.writelines(lines)
        f.close()
    else:
        np.savetxt('result/x_revised.csv', optSol, fmt='%.10f', delimiter = ',')
        f = open('result/revised_result.txt','w')
        lines = ['optimal objective value: {}\n'.format(optVal), 'number of pivots: {}\n'.format(numIter), 'total running time: {} s'.format('%.8f'%totalTime)]
        f.writelines(lines)
        f.close()


'''
util function for reading parameters
'''
def ReadParameter():
    paraList = []
    with open("./parameter.txt", "r") as f:
        data = f.readlines()
        for item in data:
            item = re.sub(r'[A-Za-z]*=','',item)
            item = re.sub(r'\n','',item)
            paraList.append(item)
    return paraList


if __name__ == "__main__":
    '''
    data preparation
    '''
    path, mode, cleanResult, method = ReadParameter()
    # change the path for different test
    #path = "./data/test3"
    A, b, c, x_star = RawData_Read(path)

    # set this parameter to be 'Bland' if exists degenerancy; otherwise 'default'
    #mode = 'Bland'
    
    '''
    set this parameter to be True if you want irrelevant variable to be 0, i.e.:
    c = [0,0,3]

    cleanResult = True:  x = [0,0,2]
    cleanResult = False: x = [1,0,2]
    '''

    #cleanResult = 'True'
    
    # ================================================================
    '''
    let b > 0
    '''
    for i in range(len(b)):
        if b[i] < 0:
            b[i] = np.negative(b[i])
            A[i,:] = np.negative(A[i,:])
    
    startauxTime = time.perf_counter()

    Aaux, B_inverse, baux, IterCountaux, baseVarList = auxPro(A,b)

    endauxTime = time.perf_counter()

    durationauxTime = endauxTime - startauxTime
    
    print("=========================================================")
    
    if method == 'classical' or method == 'both':

        startclaTime = time.perf_counter()

        cla_result, cla_optimalVal, cla_IterCount = classicSimplex(Aaux, baux, c, baseVarList, mode)

        endclaTime = time.perf_counter()

        durationclaTime = endclaTime-startclaTime
        
        if cleanResult == 'True':
            for i in range(len(c)):
                if c[i] == 0:
                    cla_result[i] = 0
        print("=========================================================")
        print("Total classical iteration : {}".format(cla_IterCount+IterCountaux))
        print("Total classical time : {}".format(durationauxTime+durationclaTime))
        WriteResult(cla_result, cla_optimalVal, cla_IterCount+IterCountaux, durationauxTime+durationclaTime,met='classical')
        print('--------------------------------------------------------------')
    # ====================================================
    if method == 'revised' or method == 'both':
        startrevTime = time.perf_counter()

        rev_result, rev_optimalVal, rev_IterCount = revSimplex(Aaux, baux, c, B_inverse, baseVarList, mode)
        
        endrevTime = time.perf_counter()

        durationrevTime = endrevTime - startrevTime
        if cleanResult == 'True':
            for i in range(len(c)):
                if c[i] == 0:
                    rev_result[i] = 0
        print("=========================================================")
        print("Total revised iteration : {}".format(rev_IterCount+IterCountaux))
        print("Total revised time : {}".format(durationauxTime+durationrevTime))
        WriteResult(rev_result, rev_optimalVal, rev_IterCount+IterCountaux,durationauxTime+durationrevTime,met='revised')
        print('--------------------------------------------------------------')
    
    print("=========================================================")

    if method == 'classical' or method == 'both':
        print("Classical calculated solution: {}".format(cla_result))
        print("Classical calculated optimal: {}".format(cla_optimalVal))
    if method == 'revised' or method == 'both':
        print("Revised calculated solution: {}".format(rev_result))
        print("Revised calculated optimal: {}".format(rev_optimalVal))

    print("...........................................")

    print("True solution: {}".format(x_star))
    print("True optimal: {}".format(np.matmul(c, x_star)))

    


    