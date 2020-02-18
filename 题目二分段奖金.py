''''
时间：2019.11.7
姓名：车立威
2017200503020
题目名称：题目二分段奖金
'''
profit_in=input("请输入利润：")
profit=int(profit_in)

bonus=[]


stage=[1000000,600000,400000,200000,100000,0]
percent=[0.01,0.015,0.03,0.05,0.075,0.1]
for i in range(6):
    if profit > stage[i]:
        bonus.append((profit-stage[i])*percent[i])
        profit=stage[i]

print("总奖金：", sum(bonus))



