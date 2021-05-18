import numpy as np
import copy
import time
import re

"""
Classic simplex
"""
def classicSimplex(A_hat, b_hat, c_hat, baseVarList_hat, mode='default'):
    A = copy.deepcopy(A_hat)
    b = copy.deepcopy(b_hat)
    c = copy.deepcopy(c_hat)
    baseVarList = copy.deepcopy(baseVarList_hat)
    
    totalIterationCount = 0

    numPrecise = 1e-10
    
    # 既约费用
    resVec = c

    # simplex问题目标函数值
    valFun = 0

    for i in range(len(baseVarList)):
        j = baseVarList[i]
        valFun -= resVec[j] * b[i]
        resVec -= resVec[j] * A[i,:]
    
    count = 0
    while True:
        # 若 rj >= 0
        if min(resVec) >= 0 or abs(min(resVec)) <= numPrecise:
            print("Classical simplex -- ends!")
            break

        totalIterationCount += 1

        minCol_id, minRow_id = 0, 0

        if mode == 'default':
            # 选出最小负既约费用系数
            minCol_id = np.argmin(resVec)

            assert max(A[:,minCol_id]) > 0, 'Problem no boundary!'
            
            # 最小正比率
            b_temp = np.ones(len(b)) * float('inf')
            for i in range(len(b)):
                if A[i,minCol_id] > numPrecise:
                    b_temp[i] = b[i] / A[i,minCol_id]
            list_btemp = b_temp.tolist()
            minRow_id = list_btemp.index(min(list_btemp))
        
        elif mode == 'Bland':
            # 选出指标最小的负既约费用系数
            for i in range(len(resVec)):
                if resVec[i] < 0 and abs(resVec[i]) > numPrecise:
                   minCol_id = i 
                   break
            
            assert max(A[:,minCol_id]) > 0, 'Problem no boundary!'

            # 最小正比率
            b_temp = np.ones(len(b)) * float('inf')
            for i in range(len(b)):
                if A[i,minCol_id] > numPrecise:
                    b_temp[i] = b[i] / A[i,minCol_id]
            minRow_id = np.argmin(b_temp)
            
        
        b[minRow_id] = b[minRow_id] / A[minRow_id, minCol_id]
        A[minRow_id,:] = A[minRow_id,:] / A[minRow_id, minCol_id]
        
        for rows in range(A.shape[0]):
            if rows != minRow_id:
                b[rows] -= A[rows,minCol_id] * b[minRow_id]
                A[rows,:] -= A[rows,minCol_id] * A[minRow_id,:]
        
        valFun -= resVec[minCol_id] * b[minRow_id]
        resVec -= resVec[minCol_id] * A[minRow_id,:]
        baseVarList[minRow_id] = minCol_id

        count += 1
        assert count <= 10000, 'Problem infinity, may exists degenerancy'

    print("Classical simplex -- optimal val: {}".format(valFun))
    print("Classical simplex -- iteration count: {}".format(totalIterationCount))
    
    x_result = np.zeros(A.shape[1])
    for i in range(len(baseVarList)):
        # if c_hat[baseVarList[i]] != 0:
        x_result[baseVarList[i]] = b[i]
    valFun = -valFun

    return x_result, valFun, totalIterationCount