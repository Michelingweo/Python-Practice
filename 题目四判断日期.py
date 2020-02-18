''''
时间：2019.11.7
姓名：车立威
2017200503020
题目名称：题目四判断日期
'''
import datetime

y = input("四位年份：")
m = input("月份：")
d = input("哪一天：")

Day = datetime.date(int(y), int(m), int(d))1
Count = Day - datetime.date(Day.year - 1, 12, 31)
print('%s是%s年的第%s天。'% (Day, y, Count.days))