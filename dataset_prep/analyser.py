import sys
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import os 


def def_value():
    return 0

def def_list():
    return []

def def_dict():
    return {'sentences':[],'factlists':[],'translated_sentences':[],'sections':[]}


filename = sys.argv[1]
lang = sys.argv[2]
folder = sys.argv[3]

jsfile = open(filename)


count_dict = defaultdict(def_value)
order_dict = defaultdict(def_list)
bbdict = defaultdict(def_dict)



totcount = 0 

for line in jsfile.readlines():
    line_dict = json.loads(line)
    qid = line_dict['qid']
    if lang == 'en':
        section = line_dict['native_sentence_section'].lower()
    else :
        section = line_dict['translated_sentence_section'].lower()

    index  = line_dict['sent_index'] 
    order_dict[qid].append(index)
    count_dict[qid] += 1 
    totcount += 1 


    bbdict[qid]['sentences'].append(line_dict['sentence'])
    bbdict[qid]['translated_sentences'].append(line_dict['translated_sentence'])
    bbdict[qid]['sections'].append(section)
    bbdict[qid]['factlists'].append(line_dict['facts'])


#print (count_dict)
    

freq_dict = defaultdict(def_value)
big_sent_count = 0


global_para_count = 0

for i in count_dict :
    freq_dict[count_dict[i]]+=1 
    
    index_list = sorted(order_dict[i]) 
    
    para_count = 0 

    curr_var =  1 
    prev = index_list[0]
    for itr in range(1,len(index_list)):
        
        if index_list[itr] - prev == 1 :
            curr_var += 1

        else :
            if curr_var >= 3 :
                para_count += 1 
                global_para_count += 1
                big_sent_count += curr_var
            curr_var = 1

        prev = index_list[itr]



print (freq_dict)


totsum = 0
sentsum  = 0

for i in freq_dict:
    if i >=3 :
        totsum += freq_dict[i]
        sentsum += freq_dict[i]*i


print ("----- "+ lang + " -----" )
print (totsum,"/", len(count_dict) )
print (sentsum, "/", totcount)
print ("***** ", global_para_count, big_sent_count, " ****")


plt.bar(list(freq_dict.keys()), freq_dict.values(), color='g')


if not os.path.exists(folder):
    os.mkdir(folder)

plt.savefig(folder + '/' +lang+'.png')


outfile = open(folder + '/' + lang+ '_parawise.jsonl','w')


for i in bbdict:
    if len(bbdict[i]['sentences']) >= 3:
        newdict = bbdict[i]
        newdict['qid'] =  i
        newstr = json.dumps(newdict)
        outfile.write(newstr+'\n')



