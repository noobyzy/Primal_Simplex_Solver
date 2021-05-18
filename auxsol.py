import numpy as np
import copy
import time
import re


'''
solve auxiliary problem
'''
def auxPro(A_hat, b_hat):
    A = copy.deepcopy(A_hat)
    b = copy.deepcopy(b_hat)

    part1IterationCount = 0

    numPrecise = 1e-10

    # 记录基变量所处的 col index: a[i]=k 第k列的第i个entry为1，其余均为0
    baseVarList = np.zeros(A.shape[0], dtype=int)
    for i in range(len(baseVarList)):
        baseVarList[i] = i + A.shape[1]

    """
    ==============================================================================
    第一部分， 判断原问题有无可行解
    """
    # 添加人工变量
    auxMat = np.eye(len(b))
    A = np.hstack((A,auxMat))
    
    # 既约费用
    resVec = np.zeros(A.shape[1])
    resVec[A_hat.shape[1]:] = np.ones(len(resVec[A_hat.shape[1]:]))

    # 辅助问题目标函数值
    valFun = 0

    for i in range(auxMat.shape[1]):
        resVec -= A[i,:]
        valFun -= b[i]
    
    count = 0
    while True:
        # 若 rj >= 0
        if min(resVec) >= 0 or abs(min(resVec)) <= numPrecise:
            print("Auxiliary problem: part I -- ends!")
            break

        part1IterationCount += 1
        # 选出最小既约费用系数 
        minCol_id = np.argmin(resVec)

        assert max(A[:,minCol_id]) > 0, 'Problem no boundary!'
        
        # 最小指标
        b_temp = np.ones(len(b)) * float('inf')
        for i in range(len(b)):
            if A[i,minCol_id] > 0:
                b_temp[i] = b[i] / A[i,minCol_id]
        minRow_id = np.argmin(b_temp)
        
        b[minRow_id] = b[minRow_id] / A[minRow_id, minCol_id]
        A[minRow_id,:] = A[minRow_id,:] / A[minRow_id, minCol_id]
        
        for rows in range(A.shape[0]):
            if rows != minRow_id:
                b[rows] -= A[rows,minCol_id] * b[minRow_id]
                A[rows,:] -= A[rows,minCol_id] * A[minRow_id,:]
        
        baseVarList[minRow_id] = minCol_id

        valFun -= resVec[minCol_id] * b[minRow_id]
        resVec -= resVec[minCol_id] * A[minRow_id,:]

        count += 1
        assert count <= 10000, 'Problem infinity, may exists degenerancy'
            

    
    print("Auxiliary problem: part I -- optimal val: {}".format(valFun))
    print("Auxiliary problem: part I -- iteration count: {}".format(part1IterationCount))

    assert abs(valFun) < numPrecise, 'Primal problem has no feasible solution'

    """
    ==============================================================================
    第二部分， 去除冗余约束，找到BFS及其对应规范形
    """

    part2IterationCount = 0
    removed_idList = []
    for i in range(len(baseVarList)):
        # 基变量中有人工变量
        if baseVarList[i] >= A_hat.shape[1]:
            print("Auxiliary var in base var: the {}th base var is the {}th aux var".format(i, baseVarList[i]-A_hat.shape[1]))
            if abs(np.max(A[i,:A_hat.shape[1]])) <= numPrecise and abs(np.min(A[i,:A_hat.shape[1]])) <= numPrecise:
                # 第 i 个约束冗余
                print("The {}th constraint is redundant, removed!".format(i))
                A = np.delete(A, i, 0)
                b = np.delete(b, i)
                removed_idList.append(i)
            else:
                # 选择index最小的非0元进行转轴
                for j in range(len(A[i,:A_hat.shape[1]])):
                    if abs(A[i,j]) > numPrecise:
                        print("The {}th base var is now in col {}".format(i,j))
                        
                        part2IterationCount += 1

                        b[i] = b[i] / A[i, j]
                        A[i,:] = A[i,:] / A[i, j]

                        for rows in range(A.shape[0]):
                            if rows != i:
                                b[rows] -= A[rows,j] * b[i]
                                A[rows,:] -= A[rows,j] * A[i,:]
                        
                        baseVarList[i] = j
                        valFun -= resVec[j] * b[i]
                        resVec -= resVec[j] * A[i,:]
                        break

    
    baseVarList = np.delete(baseVarList, removed_idList)

    print("Auxiliary problem: part II -- iteration count: {}".format(part2IterationCount))

    return A[:,:A_hat.shape[1]], np.linalg.inv(A[:,baseVarList]), b, part1IterationCount+part2IterationCount, baseVarList
