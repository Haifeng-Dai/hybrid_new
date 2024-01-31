from copy import deepcopy

a = [i for i in range(2)]
b = {cid: deepcopy(a) for cid in range(3)}
print('b', b)
a[0] = 5
print('a', a, 'b', b)