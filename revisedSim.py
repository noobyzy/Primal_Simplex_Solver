import numpy as np
import copy
import time
import re

"""
Revised simplex
"""
def revSimplex(A_hat, b_hat, c_hat, B_inverse_hat, baseVarList_hat, mode='default'):
    A = copy.deepcopy(A_hat)
    b = copy.deepcopy(b_hat)
    c = copy.deepcopy(c_hat)
    B_inverse = copy.deepcopy(B_inverse_hat)
    baseVarList = copy.deepcopy(baseVarList_hat)

    nonBaseVarList = []
    for i in range(A.shape[1]):
        if i not in baseVarList:
            nonBaseVarList.append(i)

    b_line = np.matmul(B_inverse, b)

    c_n = np.zeros(len(nonBaseVarList))
    for i in range(len(nonBaseVarList)):
        c_n[i] = c[nonBaseVarList[i]]

    c_b = np.zeros(len(baseVarList))
    for i in range(len(baseVarList)):
        c_b[i] = c[baseVarList[i]]
        
    mlambda = np.matmul(c_b, B_inverse)

    N = np.zeros((A.shape[0], len(nonBaseVarList)))
    for i in range(len(nonBaseVarList)):
        N[:,i] = copy.deepcopy(A[:,nonBaseVarList[i]])
    
    totalIterationCount = 0
    numPrecise = 1e-10
    count = 0
    while True:
        
        # 步1： 计算既约费用
        r = c_n - np.matmul(mlambda, N)
        if min(r) >= 0 or abs(min(r)) <= numPrecise:
            print("Revised simplex -- ends!")
            break
        
        totalIterationCount += 1
        if mode == 'default':
            # 步2： 选取负既约费用, 最负原则
            for i in range(len(r)):
                if not (r[i] < 0 and abs(r[i]) > numPrecise):
                    r[i] = float('inf')

            rq = np.min(r)
            q = nonBaseVarList[np.argmin(r)]
            
            # 步3： 选取最小比值， 最小原则
            yq = np.matmul(B_inverse, A[:,q])
            
            assert max(yq) > 0, 'Problem no boundary!'

            b_temp = np.ones(len(b_line)) * float('inf')
            for i in range(len(b_line)):
                if yq[i] > numPrecise:
                    b_temp[i] = b_line[i] / yq[i]
            p = np.argmin(b_temp)
                    
                    
                    

        elif mode == 'Bland':
            # 步2： 选取负既约费用, 最小指标原则
            for i in range(len(r)):
                if r[i] < 0 and abs(r[i]) > numPrecise:
                    q = nonBaseVarList[i]
                    rq = r[i] 
                    break
        
            # 步3： 选取最小比值， 最大指标原则
            yq = np.matmul(B_inverse, A[:,q])
            
            assert max(yq) > 0, 'Problem no boundary!'

            b_temp = np.ones(len(b_line)) * float('inf')
            for i in range(len(b_line)):
                if yq[i] > numPrecise:
                    b_temp[i] = b_line[i] / yq[i]
                    
            p = np.argmin(b_temp)
            # list_btemp = b_temp.tolist()
            # p = list_btemp.index(min(list_btemp))
        
        # 步4：更新
        
        mlambda += rq/yq[p] * B_inverse[p,:]
        
        Epq = np.eye(B_inverse.shape[1])
        for i in range(B_inverse.shape[1]):
            if i != p:
                Epq[i,p] = np.negative(yq[i]/yq[p])
            else:
                Epq[i,p] = 1.0/yq[p]
        B_inverse = np.matmul(Epq, B_inverse)

        nonBaseVarList.remove(q)
        nonBaseVarList.append(baseVarList[p])
        baseVarList[p] = q
        nonBaseVarList.sort()


        b_line = np.matmul(B_inverse, b)

        c_n = np.zeros(len(nonBaseVarList))
        for i in range(len(nonBaseVarList)):
            c_n[i] = c[nonBaseVarList[i]]
            
        N = np.zeros((A.shape[0], len(nonBaseVarList)))
        for i in range(len(nonBaseVarList)):
            N[:,i] = copy.deepcopy(A[:,nonBaseVarList[i]])

        count += 1
        assert count <= 20, 'Problem infinity, may exists degenerancy'
        

    x_result = np.zeros(A.shape[1])
    for i in range(len(baseVarList)):
        # if c_hat[baseVarList[i]] != 0:
        x_result[baseVarList[i]] = b_line[i]
    
    valFun = np.matmul(x_result, c)

    print("Revised simplex -- optimal val: {}".format(valFun))
    print("Revised simplex -- iteration count: {}".format(totalIterationCount))

    return x_result, valFun, totalIterationCount