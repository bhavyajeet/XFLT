
# This script checks the probablity of two facts co-occuring in a sentence given that both of them are present for the given entity  

import json
import numpy as np
import sys 
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

jsfile = open(sys.argv[1])


def def_value():
    return 0
      
countdict  = defaultdict(def_value)

jslist = jsfile.readlines()

for line in jslist:
    line_dict = json.loads(line)
    factlist = line_dict['factlists']

    flatlist = [item[0] for sublist in factlist for item in sublist]

    for i in flatlist:
        countdict[i] += 1 

    

sorted_dict =  dict(sorted(countdict.items(), key=lambda item: item[1], reverse =True ))

 
#print(sorted_dict)

#print (countdict)


count = sys.argv[2]
counter = 0 

factarr = []

for fact in sorted_dict:    
    factarr.append(fact)
    counter += 1
    if counter >= int(count):
        break


combin = list(combinations(factarr, 2))

#print (combin)


totlist = []
togetherlist = [] 



def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def is_together(fact1,allfacts):
    for ism in allfacts:
        relns = [x[0] for x in ism]
        #print (allfacts)
        #print (relns,fact1)
        if len(intersection(fact1,relns)) == len(fact1):
            return True 
    return False 
 

for i in combin:
    totval = 0
    togetherval = 0

    for line in tqdm(jslist):
        #print ('--'*10)
        line_dict = json.loads(line)
        factlist = line_dict['factlists']
        lin_flatlist = [item[0] for sublist in factlist for item in sublist]

        if len(intersection(i,lin_flatlist)) == len(i):
            totval += 1 
            if (is_together(i,factlist)):
                togetherval += 1

    totlist.append(totval)
    togetherlist.append(togetherval)


averagelist = []
nonzerolist = []

for i in range(len(combin)):
    if totlist[i] != 0 :
        print (combin[i], togetherlist[i]/totlist[i])
        averagelist.append(togetherlist[i]/totlist[i])
        if togetherlist[i]  != 0 :
            nonzerolist.append(togetherlist[i]/totlist[i])

print (np.mean(averagelist))
print (np.mean(nonzerolist))




