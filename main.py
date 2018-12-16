
l=[0,10,32,55,21,38,89,10000,273,199999,87877232]
##选择排序法
def paixu(l):
    for i in range(len(l)-1):
        for j in range(i+1,len(l)):
            if l[i]>l[j]:
                t=l[i]
                l[i]=l[j]
                l[j]=t

###冒泡排序法

def maopao(l):
    for i in range(len(l)-1):
        for j in range(len(l)-1-i):
            if l[j]>l[j+1]:
                t=l[j];
                l[j]=l[j+1]
                l[j+1]=t

maopao(l) #paixu(l)
print(l)                



   













