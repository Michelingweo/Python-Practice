''''
时间：2019.11.7
姓名：车立威
2017200503020
题目名称：题目一1234组合
'''

count = 0

for hd in range(1,5):
    for tens in range(1,5):
        for unit in range(1,5):
            if(hd!= tens) and (tens != unit) and (hd != unit):
                print('%d%d%d'%(hd, tens, unit),end=' ')
                count+=1

print("\nThe total number is:",count)
