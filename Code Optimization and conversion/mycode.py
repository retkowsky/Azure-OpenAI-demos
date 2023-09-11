# Add three to all list members.
a = [3, 4, 5]
b = a                     # a and b refer to the same list object

for i in range(len(a)):
    a[i] += 3             # b[i] also changes