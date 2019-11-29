import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

def short(x):
    if x < 0.6:
        return -(x/0.6)+1
    else:
        return 0

def medium(x):
    if x < 0.6:
        return x/0.6
    else:
        return (x-1)/(0.6-1)

def tall(x):
    if x > 0.6:
        return (x-0.6)/(1-0.6)
    else:
        return 0

# Returns a list where every element is the probability of each species
def rule_eval(SL, SW, PL, PW):
    rule1 = min(min(min( max(SL[0], SL[2]), max(SW[1], SW[2] )), max(PL[1], PL[2])), PW[1])
    rule2 = min(max(PL[0], PL[1]), PW[0])
    rule3 = min(min(max(SW[0], SW[1]), PL[2]), PW[2])
    rule4 = min(min(min(SL[1], max(SW[0], SW[1])), PL[0]), PW[2])

    return [rule2, rule1 + rule4, rule3]

correct_gusses = 0

for x, y in enumerate(iris['data']):
    
    inp1 = y[0]
    inp2 = y[1]
    inp3 = y[2]
    inp4 = y[3]

    an = ((inp1 - 4.3) / (7.9 - 4.3))
    bn = ((inp2 - 2) / (4.4 - 2))
    cn = ((inp3 - 1) / (6.9 - 1))
    dn = ((inp4 - 0.1) / (2.50 - 0.1))

    SL = [short(an), medium(an), tall(an)]
    SW = [short(bn), medium(bn), tall(bn)]
    PL = [short(cn), medium(cn), tall(cn)]
    PW = [short(dn), medium(dn), tall(dn)]

    result = rule_eval(SL, SW, PL, PW)

    species = result.index(max(result))
    if species == 0:
        print(x, '\tSetosa\t\t', iris['target'][x], '\t', result)

    elif species == 1:
        print(x, '\tVersicolor\t', iris['target'][x], '\t', result)

    elif species == 2:
        print(x, '\tVirginica\t', iris['target'][x], '\t', result)

    if species == iris['target'][x]:
        correct_gusses = correct_gusses + 1

print('Predictability: ', correct_gusses/150)
