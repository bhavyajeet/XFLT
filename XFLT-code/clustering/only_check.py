import json
import copy
lolol = 0
import re


global globalcount 
globalcount = 0

def make_facts(cluster):
    global globalcount 
    to_return = []
    #print (cluster)
    clust_list = cluster.split('<R>')
    clust_list = [i for i in clust_list if i]
    #print ('--'*10)
    #print (cluster)
    for i in clust_list:
        if i.strip() :
            try:
                thisrel = []
                head = i.split('<T>')[0].strip()
                #print (i)
                tail = i.split('<T>')[1].strip()
            except:
                globalcount += 1 
                tail = ''
                #print ("lol")

            tail_list = tail.split('<QR>')
            #no qualifiers
            quallist = []
            if len(tail_list) == 1:
                tail = tail_list[0]
            else:
                tail = tail_list[0]
                for qual in tail_list[1:]:
                    #print (qual)
                    try:
                        qr = qual.split('<QT>')[0].strip()
                        qt = qual.split('<QT>')[1].strip()
                        #print (qr)
                        #print (qt)
                        quallist.append([qr,qt])
                    except :
                        globalcount += 1 
                        qr = qual.split('<QT>')[0].strip()
                        qt = ''
                        quallist.append([qr,qt])
                        print ("lol")

            finfact = [head,tail,quallist,False]
    
            to_return.append(finfact)

    print (globalcount)
    return to_return 


langdict = {
        'as':'assamese',
        'bn':'bengali',
        'en':'english',
        'gu':'gujarati',
        'hi':'hindi',
        'kn':'kannada',
        'ml':'malayalam',
        'mr':'marathi',
        'or':'odia',
        'pa':'punjabi',
        'ta':'tamil',
        'te':'telugu'
        }

srclines = open('test-src.txt').readlines()
reflines = open('test-ref.txt').readlines()
predlines = open('test-predicted-epoch-7.txt').readlines()
ogtest = open('test.jsonl').readlines()


tuplist = []
problist = []
for linenum in range(len(srclines)):
    line = srclines[linenum]
    refline = reflines[linenum]

    numsent = len(refline.split('<BR>')[:-1])
    numfacts = len(refline.split('<R>')[1:])


    lang = line.split()[1].strip().lower()
    entity = line.split('<R>')[0].split('<H>')[1].strip()
    entity = ' '.join(entity.split())
    entity = re.sub(r'[^\x00-\x7F]+','', entity)
    idset = (lang,entity,numsent,numfacts)
    if idset in tuplist:
        print (idset)
        problist.append(idset)
    else :
        tuplist.append(idset)

print('--'*10)
big_index = {}
big_index_coll = {}


testlist = []
for line in ogtest:
    line = json.loads(line)
    lang = langdict[line['lang']]
    entity = line['entity_name'].strip().lower()
    entity = ' '.join(entity.split())
    entity = re.sub(r'[^\x00-\x7F]+','', entity)
    allfacts = line['facts_list']
    nextlist = []
    for i in allfacts:
        nextlist += i
    numfacts = len(nextlist)
    numsent = len(line['sentence_list'])
    idset = (lang,entity,numsent,numfacts)

    big_index[idset] = line
    big_index_coll[(lang,entity)] = line
    
    if idset in testlist:
        print (idset)
        problist.append(idset)
    else :
        testlist.append(idset)

print (set(tuplist)==set(testlist))



count = 0 
for i in tuplist:
    if i not in testlist :
        print (i)
        count += 1

print (count,len(tuplist))
count = 0 

for i in testlist:
    if i not in tuplist :
        print (i)
        count += 1 


print (count,len(testlist))
newlist = []

number = 0 

for num in range(len(ogtest)):
    line = ogtest[num]
    line = json.loads(line)
    lang = langdict[line['lang']]
    entity = line['entity_name'].strip().lower()
    entity = ' '.join(entity.split())
    entity = re.sub(r'[^\x00-\x7F]+','', entity)
    allfacts = line['facts_list']
    nextlist = []
    for i in allfacts:
        nextlist += i
    numfacts = len(nextlist)
    numsent = len(line['sentence_list'])
    idset = (lang,entity,numsent,numfacts)

    pred = predlines[num]
    pred = pred.replace('<pad>','')
    pred = pred.split('</s>')[0]

    clusters = pred.strip().split('<BR>')
    clusters = [k for k in clusters if k]


    srcline = srclines[num]
    refline = reflines[num]

    numsent = len(refline.split('<BR>')[:-1])
    numfacts = len(refline.split('<R>')[1:])


    lang = srcline.split()[1].strip().lower()
    entity = srcline.split('<R>')[0].split('<H>')[1].strip()
    entity = re.sub(r'[^\x00-\x7F]+','', entity)
    entity = ' '.join(entity.split())
    idset = (lang,entity,numsent,numfacts)


    #print ('-'*15) 
    #print (idset)
    #print (clusters)
    try:
        #print (big_index[idset])
        main_obj  =  big_index[idset]
    except :
        lolol += 1 
        #print (big_index_coll[(lang,entity)])
        main_obj  =  big_index_coll[(lang,entity)]

    for clust in clusters:
        number += 1 
        temp_obj = copy.deepcopy(main_obj)
        fact_obj = make_facts(clust)
        temp_obj['facts'] = fact_obj
        newlist.append(temp_obj)
        print (fact_obj)
        print (temp_obj)


print (newlist[-1])
print (newlist[-2])
#print (len(newlist))

#print (lolol)

bigstr = ''

for i in newlist:
    bigstr += json.dumps(i,ensure_ascii = False)
    bigstr += '\n'

finfile = open('test_complete.jsonl','w')
finfile.write(bigstr)








