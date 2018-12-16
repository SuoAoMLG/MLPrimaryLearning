#!/usr/bin/python3

# 环境信息：
# 
        # Ubuntu_18.3
        # VScode
        # Python 3.6
# 

# 模块引用
# 
import random
#   

# 函数定义
# shell排序
# 
def Shell_Sort(list,n):
    len = n
    # 组数初定义
    Judge_Flag = len
    while(Judge_Flag >= 1):
        # 组数减半
        # 算了不写注释了
        # 变量都没啥实际意义，懒得编名了，直接随机命名
        # 反正高组数情况下分组插排，低组数整体插排
        # over
        Judge_Flag = int(Judge_Flag/2)
        for i in range(Judge_Flag,len):
            t = list[i]
            j = i - Judge_Flag
            while(j >= 0 and  t<list[j] ):
                list[j+Judge_Flag] = list[j]
                j = j - Judge_Flag
            list[j+Judge_Flag] = t
# 

#非严格快排排序至高序再利用插排在高序下进行修正（主要是写的快排有点BUG）
# （算了，BUG找出来了。直接快排了）
def Quick_Sort(list,left,right):
    Left_Temp = left
    Right_Temp = right
    Devision = list[int((left+right)/2)]
    while(Left_Temp <= Right_Temp):
        # 与二分值比较
        # 
        while(list[Left_Temp] < Devision):
            Left_Temp += 1 
        while(list[Right_Temp] > Devision):
            Right_Temp -= 1
        # 
        # 有效交换区域
        # 
        if(Left_Temp <= Right_Temp):
            t = list[Left_Temp]
            list[Left_Temp] = list[Right_Temp]
            list[Right_Temp] = t
            Right_Temp -= 1
            Left_Temp += 1
        # 
    if(Left_Temp == Right_Temp):
        Left_Temp += 1
    # 递归调用
    # 
    if(left<Right_Temp):
        Quick_Sort(list,left,Left_Temp-1)
    if(Left_Temp<right):
        Quick_Sort(list,Right_Temp+1,right)
    # 
# 
  
# 主函数
# 随机生成 list 长度：
Length_Of_List = random.randint(15,50)
# 随机生成list中的值：
list = [random.randint(0,999) for i in range(Length_Of_List)]
print(list)
# 快排
Quick_Sort(list,0,Length_Of_List-1)
print(list)
# shell排序
Shell_Sort(list,Length_Of_List-1)
print(list)

