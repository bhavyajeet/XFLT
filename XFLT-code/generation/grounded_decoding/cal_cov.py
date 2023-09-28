import json 
import numpy as np 

lang = 'as'

lol = open(lang+'/train.jsonl').readlines()

biglist= []
bignum = []

for i in lol:
    i = json.loads(i)
    bignum.append(float(i['avg_coverage']))
    biglist += i['coverage_score_list']


finbig = [float(x) for x in biglist]

print(np.percentile(biglist, 33))
print(np.percentile(biglist, 66))


