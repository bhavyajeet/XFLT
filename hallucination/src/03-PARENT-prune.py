import json
import re 
import sys 


lang  = sys.argv[1]
thresh  = float(sys.argv[2])


filen = '/scratch/model_outputs/towork/results-all-euclidean/'+lang+'-coverage.jsonl'

out_ref = open('./ref-file','w')
out_og_gen = open('./og-file','w')
out_new_gen = open('./new-file','w')

lines = open(filen).readlines()

count = 0 

reflist = []
oggenlist = []
newgenlist = []


for line in lines:
    count += 1 
    jsonobj = json.loads(line.strip())
    newstr = '' 
    for i in jsonobj['scores']:
        if i[0] not in  ['[SEP]','[CLS]']:
            if i[2] <= thresh:
                newstr += i[0] + ' '
    newstr = newstr.replace(' ##','')
    #print (newstr)

    reflist.append(jsonobj['ref_sentence'].strip()+'\n')
    oggenlist.append(jsonobj['sentence'].strip()+'\n')
    newgenlist.append(newstr+'\n')

    #if count >= 10:
        #break
    

out_ref.writelines(reflist)
out_og_gen.writelines(oggenlist)
out_new_gen.writelines(newgenlist)


