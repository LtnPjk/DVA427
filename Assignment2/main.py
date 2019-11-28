import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

#Inputs
sepal_length = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Sepal Length')
sepal_width = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Sepal Width')
petal_length = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Petal Length')
petal_width = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Petal Width')
species = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Species')

# Generate fuzzy membership functions
sepal_length['short']  = fuzz.trimf(sepal_length.universe, [0, 0, 0.6])
sepal_length['medium'] = fuzz.trimf(sepal_length.universe, [0, 0.6, 1])
sepal_length['long']   = fuzz.trimf(sepal_length.universe, [0.6, 1, 1])  

sepal_width['short']  = fuzz.trimf(sepal_width.universe, [0, 0, 0.6]) 
sepal_width['medium'] = fuzz.trimf(sepal_width.universe, [0, 0.6, 1])
sepal_width['long']   = fuzz.trimf(sepal_width.universe, [0.6, 1, 1])   

petal_length['short']  = fuzz.trimf(petal_length.universe, [0, 0, 0.6])
petal_length['medium'] = fuzz.trimf(petal_length.universe, [0, 0.6, 1])
petal_length['long']   = fuzz.trimf(petal_length.universe, [0.6, 1, 1])  

petal_width['short']  = fuzz.trimf(petal_width.universe, [0, 0, 0.6])
petal_width['medium'] = fuzz.trimf(petal_width.universe, [0, 0.6, 1])
petal_width['long']   = fuzz.trimf(petal_width.universe, [0.6, 1, 1])

species['setosa']  = fuzz.trimf(species.universe, [0, 0, 0.5])
species['versicolor'] = fuzz.trimf(species.universe, [0, 0.5, 1])
species['virginica']   = fuzz.trimf(species.universe, [0.5, 1, 1])

#species.automf(3)

species.view()
plt.show()
#sepal_length.view()

# Construct rules
rule1 = ctrl.Rule((sepal_length['short'] | sepal_length['long']) & (sepal_width['medium'] | sepal_width['long']) & (petal_length['medium'] | petal_length['long']) & petal_width['medium'], species['versicolor'])
rule2 = ctrl.Rule((petal_length['short'] | petal_length['medium']) & petal_width['short'], species['setosa'])
rule3 = ctrl.Rule((sepal_width['short'] | sepal_width['medium']) & petal_length['short'] & petal_width['long'], species['virginica'])
rule4 = ctrl.Rule(sepal_length['medium'] & (sepal_width['short'] | sepal_width['medium']) & petal_length['short'] & petal_width['long'], species['versicolor'])

species_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
species_res = ctrl.ControlSystemSimulation(species_ctrl)

correct_gusses = 0

for x, y in enumerate(iris['data']):
    
    inp1 = y[0]
    inp2 = y[1]
    inp3 = y[2]
    inp4 = y[3]

    species_res.input['Sepal Length'] = ((inp1 - 4.3) / (7.9 - 4.3))
    species_res.input['Sepal Width'] = ((inp2 - 2) / (4.4 - 2))
    species_res.input['Petal Length'] = ((inp3 - 1) / (6.9 - 1))
    species_res.input['Petal Width'] = ((inp4 - 0.1) / (2.51 - 0.1))
     
    species_res.compute()   
    
    '''
    species_res.input['Sepal Length'] = 0.3
    species_res.input['Sepal Width'] = 0.8
    species_res.input['Petal Length'] = 0.2
    species_res.input['Petal Width'] = 0.7

    species_res.compute()

    species.view(sim=species_res)

    plt.show()
    break
    '''

    if species_res.output['Species'] < 0.25:
        print(x, 'Species: Setosa, ', iris['target'][x])
        result = 0
    elif species_res.output['Species'] < 0.75:
        print(x, 'Species: Versicolor, ', iris['target'][x])
        result = 1
    else:
        print(x, 'Species: Virginica, ', iris['target'][x])
        result = 2

    if result == iris['target'][x]:
        correct_gusses = correct_gusses + 1
    '''
    if x == 99:
        break
    '''
    #print(species_res.output['Species'])

    #species.view(sim=species_res)

    #plt.show()

print('Predictability: ', correct_gusses/150)