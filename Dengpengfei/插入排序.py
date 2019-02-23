/*
*  简单的插入排序
*/
ListA = [34,8,64,51,32,21]        // 一个简单的整数元组，当然也可以是其他的元素
ListB = ListA        // 用ListB 保存原始的元组
  for i in range(len(ListA)+1):   // 遍历整个元组
      tmp = ListA[i]       // 保存遍历到的元素
      j = i                
      while j > 0 && tmp < ListA[j-1]:  // 遍历j 之前的所有元素，求得满足条件的元素
        ListA[j] = ListA[j-1]
        --j
      ListA[j] = tmp
print(ListB)   // 输出原来的元组
print(ListA)   // 输出排序后的元组
