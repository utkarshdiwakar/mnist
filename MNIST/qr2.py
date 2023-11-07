import statistics
gay = [int(x) for x in input().split(" ")]
lst1 = [int(x) for x in input().split(" ")]
lst2 = [int(x) for x in input().split(" ")]
lst3 = [int(x) for x in input().split(" ")]
lst4 = [int(x) for x in input().split(" ")]
lst5 = [int(x) for x in input().split(" ")]
lst1 = lst1 + lst2 + lst3 + lst4 + lst5
# print(lst1)
ans = statistics.mode(lst1)
print(ans)

