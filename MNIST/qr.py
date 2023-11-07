def diff_finder(lst,n):
    lst.append(lst[0])
    diff = [abs(lst[x+1]-lst[x]) for x in range(n)]
    lst.pop()
    return diff

n = int(input())
lst = [] 
lst = [int(x) for x in input().split(" ")]

x = int(input())
B = lst.copy()
for i in range (x):
    A = B.copy()
    B = diff_finder(A.copy(), n)

print("A:", *A)
print("B:", *B)
