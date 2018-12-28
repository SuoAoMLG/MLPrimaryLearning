# Bubble Sort

*依次比较相邻元素大小，并作交换*

```python
list1 = [43,1,54,76,32,76,45,89]

#check the number of program
m = 1                      

def Bubble_Sort():
	global m
	for i in range(len(list1)):
		m += 1
		for j in range(len(list1) - 1 - i):
			m += 2
			if list1[j] > list1[j + 1]:
				m += 1
				list1[j], list1[j + 1] = list1[j + 1], list1[j]

Bubble_Sort()
print(list1)
print(m)
```

 

# Selection Sorts

*依次比较相邻元素大小，但不做交换，将最大值放到最后（前）*



```python
list1 = [43,1,54,76,32,76,45,89]

#check the number of program
m = 1                      

def Selection_Sort():
    global m 
    for i in range(len(list1) - 1):
        maxindex = 0
        m += 2
        for j in range(len(list1) - 1 - i):
            m += 2
            if list1[j + 1] > list1[maxindex]:
                m += 1
                 #标记出最大的数
                maxindex = j + 1  
        #exchange
        list1[len(list1) - 1 - i],list1[maxindex] = list1[maxindex],list1[len(list1) - 1 - i]
        m += 1
        
Selection_Sort()
print(list1)
print(m)
```



# Quick Sort

*先选取一个基准数，将未排序数分为大于基准数与小于基准数的两个区，最后将基准数交换至分区的边界处，不断对数列进行分区迭代直至区的大小为1停止，排序完毕*



```python
#分区
def Partition(arr, firstindex, lastindex):
    i = firstindex - 1
    for j in range(firstindex, lastindex):
        if arr[j] <= arr[lastindex]:
            i += 1
            arr[i],arr[j] = arr[j],arr[i]
	#exchange
	arr[i + 1],arr[lastindex] = arr[lastindex],arr[i+1]
	return i




def Quick_Sort(arr, firstindex, lastindex):
    if firstindex < lastindex:
        #选取基准数arr[divindex]
        divindex = Partition(arr, firstindex, lastindex)
        Quick_Sort(arr, firstindex, divindex)
        Quick_Sort(arr, divindex+1, lastindex)


	else:
		return


list1 = [43,1,54,76,32,76,45,89]
Quick_Sort(list1,0,len(list1)-1)
print(list1)
```

