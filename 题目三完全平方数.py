''''
时间：2019.11.7
姓名：车立威
2017200503020
题目名称：题目三完全平方数
'''

import math
#假设该数为10000以内正整数
for i in range(0,10000):
    count = 0
    plus1 = i + 100
    st1 = math.sqrt(plus1)
    plus2 = i + 268
    st2 = math.sqrt(plus2)

    if st1 - math.floor(st1) == 0 and st2 - math.floor(st2) == 0:
        print(i)


#注意math.sqrt的返回结果是float，即便是整数也不能用isinstance（）判断是整型

