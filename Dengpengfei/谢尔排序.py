/**
*     谢尔排序
*/
def shellsort(ListA):
      gap = len(ListA)/2
      while gap > 0:
         for i in range(gap,len(ListA)):
             tmp = ListA[i]
             j = i
             while j>=gap && tmp < ListA[j-gap]:
                 a[j] = a[j-gap]
                 j = j-gap
            a[j] = tmp
     