Presented by Yuzy 2018533124
==========================

## main.py
Classical Simplex 方法线性求解器，直接运行即可. 参数在 parameter.txt 中修改

## auxsol.py
辅助问题第一阶段求解

## classicalSim.py
classical simplex 求解

## revisedSim.py
revised simplex 求解

## draw.py
绘图

## data

存储输入数据

## report

报告

## result

每次运行后更新此处

* x_simplex.csv 计算得到的optimal solution

* result.txt 计算得到的目标函数最优值，转轴次数，总耗时

## parameter.txt

* path 在‘=’后更改输入数据路径

* pivotMode 决定了转轴规则。在非退化情况下，可使用default; 否则请使用 Bland

>> default

>> Bland

* cleanResult 由于部分变量对目标函数无影响，可以使用cleanResult将其置为0

i.e. c = [0,0,3]

    cleanResult = True:  x = [0,0,2]
    cleanResult = False: x = [1,0,2]

>> True

>> False

* method 决定使用 classical或revised或两者都用

>> classical

>> revised

>> both