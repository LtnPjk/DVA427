from sklearn.datasets import load_iris

iris = load_iris()

'''
for x, y in iris.items():
    print(x,y)
'''

print(iris['data'][1][1])
print(iris['target'][1])

for x, y in enumerate(iris['data']):

    print(y[1])

for a, b in iris.items():
    print(a, b)