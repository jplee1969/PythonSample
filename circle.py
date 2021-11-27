n=int(input("请输入总人数：n="))
k=int(input("请规定报到数字几的人退出圈子：k="))
circle=list(range(1,n+1))
print("游戏开始前的初始位置",circle)
num=1
while len(circle)!=1:
    circle.append(circle.pop(0)) #把已报数的人取出放到队尾，以此实现围成圈循环往复
    num+=1
    if num==k:
        circle.pop(0) #把报到规定数字的人踢出圈子
        num=1 #重新从1开始报数
        print("剩余的列表是：",circle)
print("最后留下的人是原来第{}号的人".format(*circle))