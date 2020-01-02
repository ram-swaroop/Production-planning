"""
Created on Thu Jun 27 16:59:45 2019

@author: JAE6KOR
"""
import time
st = time.time()
import os
import numpy as np
import pandas as pd
os.chdir(r'D:\Bosch Hackathon\Aggregate planning') # setting working directory


'Required input data'
product_details = pd.read_csv('product_details.csv')
demand_forecast = pd.read_csv('demand_forecast.csv').T

"Steps to put demand_forecast in convenient dataframe"
new_header = demand_forecast.iloc[0]
demand_forecast = demand_forecast[1:]
demand_forecast.columns = new_header
demand_forecast = demand_forecast.reset_index(drop=['index'])


from pulp import * # package used to solve linear programming
prob = LpProblem("Aggregate planning",LpMinimize)



"Lists required to generate variables and further operations"
days = [str(i+1) for i in range(14)]
weeks = [str(i+1) for i in range(2)]
products = [str(i+1) for i in range(3)]
machines = [str(i+1) for i in range(14)]



'function to find the probability of a demand using triangular distribution'
def probability(x, d):
    if (x>=int(0.95*d)) & (x< int(d)):
        return 2*(x - int(0.95*d))/((int(d)-int(0.95*d))*(int(1.05*d)-int(0.95*d)))
    elif (x>=int(d)) & (x <= int(1.05*d)):
        return 2*(int(1.05*d)-x)/((int(d)-int(0.95*d))*(int(1.05*d)-int(0.95*d)))
        
    
    
"Problem variables (prod_vars, alloc_vars, setup_vars, lost_sales_vars)"
prod_vars = LpVariable.dicts("prod",(days,products,machines),0,None,LpInteger) # to know abot the quantity of production for product p by machine m on day t
alloc_vars = LpVariable.dicts("alloc",(days,products,machines),0,1,LpInteger) # to know whether a mahicne m is allocated for product p on a day t
setup_vars = LpVariable.dicts("setup",(days[1:],products,machines),0,1,LpInteger) # setup variables are used after day 1 (to know whether we should setup the machine (1) or not (0))


lost_sales_vars = [[None]*len(products) for _ in range(len(weeks))] # 2d-list to store the variables (because of uncertain demands) of product p on week w
demand_set = [[None]*len(products) for _ in range(len(weeks))] # 2d- list to store the demands (because of uncertain demands) of product p on week w 

for i in range(len(weeks)):
    for p in products:
        d = demand_forecast['P'+ p][int(i)] # forecasted demand of product p on week i
        s = list(range(int(d*.95), int(d*1.05)+1)) # range of all demands possible for product p on week i
        
        "constraints to reduce model complexity by considering selected demands from the list of all possible demands for product p on week i"
        if d<=300:
            s1 = s
        else:
            s1 = s[::int(len(s)/15)]
            
        demand_set[i][int(p)-1] = s1
        tot_lost_sales = [str(i+1) for i in range(len(s1))] # will be used to generate lost_sales_vars
        lost_sales_vars[i][int(p)-1] = LpVariable.dicts("lostsales_"+str(i)+"_"+p,(tot_lost_sales),0,None,LpInteger)



""" TI1 + TI2 + TI3 : Total inventory items of all days (inventory is counted at the end of the day (both WIP and FG items)
    TLS: total lost sales of all products and weeks
   """
TI1 = 0
TI2 = 0
TI3 = 0
TLS = 0 
for t in days:
    for p in products:
        TI1+= product_details['WIP_initial'][int(p)-1] + lpSum([lpSum([prod_vars[t1][p][m1] for m1 in machines[:8]]) for t1 in days[:int(t)]])\
        + product_details['FG_initial'][int(p)-1]
        
for w in weeks:
    for p in products: 
        s = list(range(int(demand_forecast['P'+ p][int(w)-1]*.95),int(demand_forecast['P'+ p][int(w)-1]*1.05)+1))
        dm = demand_forecast['P'+ p][int(w)-1]

        if dm<=300:
            s1 = s
        else:
            s1 = s[::int(len(s)/15)]
            
        prob_list = [probability(i, dm) for i in s1] # probability list for the considered demands of product p on week w
        TI2+= (len(weeks)+1-int(w))*lpSum([np.array([prob_list[j]*lost_sales_vars[int(w)-1][int(p)-1][str(j+1)] for j in range(len(s1))])])/sum(prob_list)
        
        TI3+= lpSum([demand_forecast['P'+ p][:int(w)]])
        
        TLS+= lpSum([np.array([prob_list[j]*lost_sales_vars[int(w)-1][int(p)-1][str(j+1)] for j in range(len(s1))])])/sum(prob_list)

"""
############# Objective function #########

"""
prob+= 0.5*(TI1 + TI2 -TI3) + 1000*TLS     



        
"""
############# Constraints #############

"""

" constraints with lost Sales, FG, and Demand"
for t in days:
    print(t)
    if t=='7':
        for p in products:
            for w1 in range(len(lost_sales_vars[int(int(t)/7-1)][int(p)-1])):
                prob+=lpSum([product_details['FG_initial'][int(p)-1] + lpSum([lpSum([prod_vars[t2][p][m2] for m2 in machines[8:14]])\
                                 for t2 in days[:int(t)]]) + lost_sales_vars[int(int(t)/7-1)][int(p)-1][str(w1+1)] - demand_set[int(int(t)/7-1)][int(p)-1][w1]]) >=0, ''
    elif t=='14':
        for p in products:
            print(p)
            for w1 in range(len(lost_sales_vars[int(int(t)/7-2)][int(p)-1])):
                for w2 in range(len(lost_sales_vars[int(int(t)/7-1)][int(p)-1])):
                    print(w1, w2)
                    prob+=lpSum([product_details['FG_initial'][int(p)-1] + lpSum([lpSum([prod_vars[t2][p][m2] for m2 in machines[8:14]])\
                                     for t2 in days[:int(t)]]) + lost_sales_vars[int(int(t)/7-2)][int(p)-1][str(w1+1)] + lost_sales_vars[int(int(t)/7-1)][int(p)-1][str(w2+1)] - demand_set[int(int(t)/7-2)][int(p)-1][w1]  - demand_set[int(int(t)/7-1)][int(p)-1][w2] ]) >=0, ''

"WIP and FG constraints (per day per product) / constraints to remove excess production"
for t in days:
    print(t)
    for p in products:
            prob+= product_details['WIP_initial'][int(p)-1] + lpSum([lpSum([prod_vars[t1][p][m1] for m1 in machines[:8]]) for t1 in days[:int(t)-1]])\
            - lpSum([lpSum([prod_vars[t2][p][m2] for m2 in machines[8:14]]) for t2 in days[:int(t)]]) >=0, ''
                           
             
'constraints to remove excess work-hours'     
for t in days:
    if t=='1':
        prob+=lpSum([lpSum([(int(m)<=8)*(60*product_details['S1_setup_time'][int(p)-1]*alloc_vars[t][p][m] + prod_vars[t][p][m]*3) + (int(m)>8)*(60*product_details['S2_setup_time'][int(p)-1]*alloc_vars[t][p][m] + prod_vars[t][p][m]*2) for m in machines]) for p in products]) - 360*60 <=0 , ''
    else:
        prob+=lpSum([lpSum([(int(m)<=8)*(60*product_details['S1_setup_time'][int(p)-1]*setup_vars[t][p][m] + prod_vars[t][p][m]*3) + (int(m)>8)*(60*product_details['S2_setup_time'][int(p)-1]*setup_vars[t][p][m] + prod_vars[t][p][m]*2) for m in machines]) for p in products]) - 360*60 <=0 , ''      
 
'constraints to remove excess machine-hours (per machine)'     
for t in days:
    for m in machines:
        if t=='1':
            prob+=lpSum([(int(m)<=8)*(60*product_details['S1_setup_time'][int(p)-1]*alloc_vars[t][p][m] + prod_vars[t][p][m]*3) + (int(m)>8)*(60*product_details['S2_setup_time'][int(p)-1]*alloc_vars[t][p][m] + prod_vars[t][p][m]*2) for p in products]) - 24*60 <=0 , ''
        else:    
            prob+= lpSum([(int(m)<=8)*(60*product_details['S1_setup_time'][int(p)-1]*setup_vars[t][p][m] + prod_vars[t][p][m]*3) +\
                          (int(m)>8)*(60*product_details['S2_setup_time'][int(p)-1]*setup_vars[t][p][m] + prod_vars[t][p][m]*2) for p in products]) - 24*60 <=0 , ''
    
    
'constraints for machine allocation'
'If a machine is allocated for  product on a given day then some production will be there otherwise zero production'
for t in days:
    for p in products:
        for m in machines:
            prob+=prod_vars[t][p][m] - alloc_vars[t][p][m]*0.1 >= 0, ''
            prob+=prod_vars[t][p][m] - alloc_vars[t][p][m]*100000 <= 0 , ''


'setup time variable constraints'
for t in days[1:]: # setup variables are used after day 1 (to know whether we should setup the machine (1) or not (0))
    for p in products:
        for m in machines:
            prob+= setup_vars[t][p][m] + alloc_vars[str(int(t)-1)][p][m] - alloc_vars[t][p][m] >= 0, ''

                        
for t in days:
    for m in machines:
        prob+=lpSum([alloc_vars[t][p][m] for p in products]) <= 1, ''

'elastic constraints'
for t in days:
    if t=='1':
        prob.extend(LpConstraint(e=LpAffineExpression(lpSum([lpSum([(int(m)<=8)*(60*product_details['S1_setup_time'][int(p)-1]*alloc_vars[t][p][m] + prod_vars[t][p][m]*3) + (int(m)>8)*(60*product_details['S2_setup_time'][int(p)-1]*alloc_vars[t][p][m] + prod_vars[t][p][m]*2) for m in machines]) for p in products])), sense=-1, name=t, rhs=240*60).makeElasticSubProblem(penalty = 0.833, proportionFreeBoundList = [0,0]))
    else:
        prob.extend(LpConstraint(e=LpAffineExpression(lpSum([lpSum([(int(m)<=8)*(60*product_details['S1_setup_time'][int(p)-1]*setup_vars[t][p][m] + prod_vars[t][p][m]*3) + (int(m)>8)*(60*product_details['S2_setup_time'][int(p)-1]*setup_vars[t][p][m] + prod_vars[t][p][m]*2) for m in machines]) for p in products])), sense=-1, name=t, rhs=240*60).makeElasticSubProblem(penalty = 0.833, proportionFreeBoundList = [0,0]))


# The problem is solved using PuLP's choice of Solver
prob.writeLP("Agg_planning_stochastic_2_weeks_21.lp")
prob.solve() # default: CBC
print('Total solution time: ',time.time() - st)

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])


# Each of the variables is printed with it's resolved optimum value
prob_var_names = []
prob_var_values = []
for v in prob.variables():
    prob_var_names+=[v.name]
    prob_var_values+=[v.varValue]
    
# The optimised objective function value is printed to the screen
print("Total production cost ", value(prob.objective))


"*alloc_data* is a dataframe to store the allocated product on machine m on day t and *prod_data* will represent the production quantity for that product"

alloc_data = pd.DataFrame(np.nan, index = list(range(len(days))), columns  = ['Day']+['M'+m for m in machines])
prod_data = pd.DataFrame(np.nan, index = list(range(len(days))), columns  = ['Day']+['M'+m for m in machines])
alloc_data['Day'] = [int(t) for t in days]
prod_data['Day'] = [int(t) for t in days]


for i in range(3*len(days),3*len(days)+len(days)*len(products)*len(machines)):
    if i%100==0:
        print(i)
    if prob_var_values[i]==1:
        s2 = prob_var_names[i].split('_')
        alloc_data['M'+s2[3]][int(s2[1])-1] = 'P'+s2[2]
        
lost_len = 0 # count of total lost sales variables
for i in range(np.shape(demand_set)[0]):
    for j in range(np.shape(demand_set)[1]):
        lost_len+=len(demand_set[i][j])
        
for i in range(3*len(days)+len(days)*len(products)*len(machines)+ lost_len,len(prob_var_names)-len(days[1:])*len(products)*len(machines)):
    if i%100==0:
        print(i)    
    if prob_var_values[i] > 0:
        s2 = prob_var_names[i].split('_')
        prod_data['M'+s2[3]][int(s2[1])-1] = prob_var_values[i]
        
#prod_data.to_csv('Prod_data_stochastic_2_weeks_2.csv')
#alloc_data.to_csv('Alloc_data_stochastic_2_weeks_2.csv')

#%%
"""TO VERIFY DEMAND CONSTRAINTS"""

'Total demand till the end of week 1'
#step1 = [0]*15
#step2 = [0]*15
#
#for i in range(len(step1)):
#    for j in range(7):
#        for k in range(8):
#            if alloc_data['M'+str(k+1)][j]=='P'+str(i+1):
#                step1[i]+=prod_data['M'+str(k+1)][j]
#        for k in range(8,14):
#            if alloc_data['M'+str(k+1)][j]=='P'+str(i+1):
#                step2[i]+=prod_data['M'+str(k+1)][j]
                
'Total demand till the end of week 2'
#step1 = [0]*15
#step2 = [0]*15
#
#for i in range(len(step1)):
#    for j in range(len(alloc_data)):
#        for k in range(8):
#            if alloc_data['M'+str(k+1)][j]=='P'+str(i+1):
#                step1[i]+=prod_data['M'+str(k+1)][j]
#        for k in range(8,14):
#            if alloc_data['M'+str(k+1)][j]=='P'+str(i+1):
#                step2[i]+=prod_data['M'+str(k+1)][j]

#%%
""" Simple optimization problem """
from pulp import *
prob = LpProblem("Opt_model_1",LpMinimize)

d1 = 100 # week 1 demand
d2 = 320 # week2 demand
x1 = LpVariable("x1",0,None,LpInteger) # production
x2 = LpVariable("x2",0,None,LpInteger)

l1 = LpVariable("l1",0,None,LpInteger) # lost sales
l2 = LpVariable("l2",0,None,LpInteger)

'opt. function'
IC1 = 2*(x1 + l1 - d1) # week1 inventory cost
IC2 = 2*((x1 + l1 - d1) + x2 + l2 - d2) # week2 inventory cost

LSC1 = 50*l1 # week1 inventory cost
LSC2 = 50*l2 # week1 inventory cost

prob+= IC1 + IC2 + LSC1 + LSC2

'constraints'
prob+= x1 + l1 >= d1, ''
prob+= (x1 + l1 - d1) + x2 + l2 >= d2, ''

prob.solve() # default solver CBC

print("Status:", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)
    
# The optimised objective function value is printed to the screen
print("Optimum prod. cost: ", value(prob.objective))

#%%
""" Stochastic optimization problem """
from pulp import *
prob = LpProblem("Opt_model_1",LpMinimize)
d11 = 100; d12 = 150 # week 1 demand
p11 = 0.4; p12 = 0.6 # probability week1 demands

d21 = 300; d22 = 320 # week2 demand
p21 = 0.3; p22 = 0.7 # probability week2 demands

x1 = LpVariable("x1",0,None,LpInteger) # production
x2 = LpVariable("x2",0,None,LpInteger)

l11 = LpVariable("l11",0,None,LpInteger)
l12 = LpVariable("l12",0,None,LpInteger)   # lost sales
l21 = LpVariable("l21",0,None,LpInteger)
l22 = LpVariable("l22",0,None,LpInteger)


'opt. function'
IC1 = 2*p11*(x1 + l11 - d11) + 2*p12*(x1 + l12 - d12) # week1 inventory cost
IC2 = 2*p21*(p11*(x1 + l11 - d11) + p12*(x1 + l12 - d12) + x2 + l21 - d21)\
         + 2*p22*(p11*(x1 + l11 - d11) + p12*(x1 + l12 - d12) + x2 + l22 - d22) # week2 inventory cost

LSC1 = 50*(p11*l11 + p12*l12) # week1 inventory cost
LSC2 = 50*(p21*l21 + p22*l22) # week1 inventory cost

prob+= IC1 + IC2 + LSC1 + LSC2

'constraints'
prob+= x1 + l11 >= d11, ''  # week1
prob+= x1 + l12 >= d12, ''

prob+= (x1 + l11 - d11) + x2 + l21 >= d21, '' # week2
prob+= (x1 + l12 - d12) + x2 + l21 >= d21, '' # week2

prob+= (x1 + l11 - d11) + x2 + l22 >= d22, '' # week2
prob+= (x1 + l12 - d12) + x2 + l22 >= d22, '' # week2

prob.solve() # default solver CBC

print("Status:", LpStatus[prob.status])

for v in prob.variables():
    print(v.name, "=", v.varValue)
    
# The optimised objective function value is printed to the screen
print("Optimum prod. cost: ", value(prob.objective))

