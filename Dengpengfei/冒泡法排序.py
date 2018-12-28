/**
*    冒泡法排序
*/
a = [12,23,5,56,56,89,52]
for i in range(len(a)):
     for j in range(i+1,len(a)):
        if a[i]>=a[j]:
             t = a[j]
             a[j]=a[i]
             a[i]=t