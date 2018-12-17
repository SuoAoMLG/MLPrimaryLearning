def lizanpaixu(arr):
    for j in range(len(arr) - 1, 0, -1):  # 如果我没猜错这个语法的意思是从最后一个数依次减一减到0...
        for i in range(0, j):             #可能从0往上加默认加一。。。  
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]   #我被秀到了

birthday = [1,9,9,9,0,2,1,5]      #嗯，我的生日
lizanpaixu(birthday)
print(birthday)  
