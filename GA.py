#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import GA_utils

## Initializing Values:
N = 50
Pc = 0.9
Pm = 0.1
ITER = 100
m = 2
BS = np.array([10,10])
L = sum(BS)
Lo = np.array([-4,-1.5])
Hi = np.array([2,1])

best_so_far = [0]
best_ans_r = None  # just init value!
Average_fitness = []

## First Generation:
Population = np.round(np.random.rand(N,L)).astype(int)

### show everythings! ###
### if you don't want to draw, comment this section:----------------------
plt.ion()
plt.figure()
#####  -------------------------------------------------------------------
## ## ## ##
for it in range(ITER):
    real_val=GA_utils.chrome_decode(Population, BS, m, Lo, Hi)
    ## Evaluation:
    fit = GA_utils.cost_function(real_val)
    
    max_fit = fit.max()
    best_ans_r = real_val[fit.argmax(), :] if best_so_far[-1] < max_fit else best_ans_r ## opt_sol in your code!
    best_so_far.append(max(max_fit, best_so_far[-1]))
    Average_fitness.append(fit.mean())
    
    ## Show everythings: --------------------------
    ### if you want speed so comment this section:
    plt.cla()
    plt.plot(real_val[:,0], real_val[:,1], '.')
    plt.plot(best_ans_r[0], best_ans_r[1], 'X')
    plt.grid()
    plt.axis([Lo[0],Hi[0], Lo[1], Hi[1]])
    plt.title('iteration = ' + str(it))
    plt.pause(0.01)
    ### -----------------------------------------
    ## Selection new Parents
    parents = GA_utils.selection(fit, Population)
    
    ## Crossover:
    children = GA_utils.crossingover(parents, Pc)
    
    ## Mutation:
    Population = GA_utils.mutation(children, Pm)

plt.ioff()
plt.cla()
plt.plot(real_val[:,0], real_val[:,1], '.')
plt.plot(best_ans_r[0], best_ans_r[1], 'X')
plt.grid()
plt.axis([Lo[0],Hi[0], Lo[1], Hi[1]])
plt.title("Last Population")
plt.legend(["Population", "best answer"])
plt.figure()
plt.plot(best_so_far[1::], 'g')
plt.plot(Average_fitness, 'r')
plt.title("fitness over iterations")
plt.legend(["best so far", "Average of fitness"])
plt.show()
print("best answer: ", best_ans_r, '\n', "best fitness: ", best_so_far[-1])
print (GA_utils.cost_function(np.array([best_ans_r])))
