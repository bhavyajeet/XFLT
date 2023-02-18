# Takes probabilities and creates val.csv file


import pandas as pd 
import json
import sys

languages = [ 'hi','en','bn','ta','te','mr','gu','kn','ml','pa','or','as' ]
folder = sys.argv[1]
#languages = ['en']
finlist = []


yarc = 0
farc = 0

for lang in languages:
    maxnum  = 0 
    maxqid = None
    maxflen = 0
    thebigsent = ''

    filename = folder + '/' + lang + '_parawise.jsonl'
    trainfile = open (filename)

    for datapt in trainfile.readlines():
        line_dict = json.loads(datapt)

        factlist = line_dict['factlists']
        
        outstr = lang 
        for ff in factlist:
            for fk in ff:
                outstr += ' <r> ' +  fk[0].lower() 


        finlist.append({
            'factlist':outstr,
            'label': len(line_dict['sentences']) 
            })

        currnum = len(line_dict['sentences'])
        if currnum == 3 :
            yarc+=1
        elif currnum ==4 :
            farc += 1


        if currnum > maxnum:
            maxnum = currnum 
            maxqid = line_dict['qid']
            thebigsent = line_dict['translated_sentences']
            flat_list = [item for sublist in line_dict['factlists'] for item in sublist]
            maxflen = len(flat_list)

    print (lang,maxnum,maxqid,maxflen)
    #print (thebigsent)

df = pd.DataFrame(finlist)
print (yarc,farc)

df.to_csv(folder + '/' + folder  + '_all.csv',index=False,header=False)
