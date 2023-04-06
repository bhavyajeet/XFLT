import json
bcb= 0
import re

biglist2 = [ 'performer' ,'depicts','composer','lyrics_by','characters','dan/kyu rank', 'related category', 'coat of arms', 'occupant', 'has pet', 'board member', 'partner in business or sport', 'victory', 'owner of', 'location of formation', 'valid in period', 'connects with', 'allegiance', 'Roman agnomen', 'date of disappearance', 'competition class', 'nominated for', 'commemorates', 'significant person', 'publisher', 'replaces', 'author', 'academic degree', 'derivative work', 'noble title', 'total goals in career', 'eye color', 'medical condition', 'archives at', 'has effect', 'total assists in career', 'coach of sports team', 'statement is subject of', 'depicted by', 'anthem', 'interested in', 'native language', 'sponsor', 'convicted of', 'work period (start)', 'title of chess person', 'Elo rating', 'commander of (DEPRECATED)', 'hair color', 'employer', 'indigenous to', 'mount', 'start time', 'part of', 'set in period', 'sibling', 'total points in career', 'charge', 'publication date', 'facet of', 'parliamentary group', 'birthday', 'headquarters location', 'school of', 'religion', 'record held', 'point in time', 'member of political party', 'ancestral home', 'represents', 'military casualty classification', 'armament', 'culture', 'style of karate', 'place of origin (Switzerland)', 'said to be the same as', 'color', 'date of death', 'has list', 'date of birth', 'character role', 'creator', 'penalty minutes in career', 'represented by', 'father', 'writing language', 'influenced by', 'Eight Banner register', 'stated in', 'contributed to creative work', 'doctoral advisor', 'name', 'location of discovery', 'social classification', 'Roman nomen gentilicium', 'genre', 'language used', 'place of burial', 'chairperson', 'nickname', 'voice type', 'cast member', 'manner of death', 'replaced by', 'height', 'parliamentary term', 'list of works', 'part of the series', 'floruit', 'work location', 'discography', 'possible creator', 'director', 'end time', 'studies', 'killed by', 'relative', 'movement', 'participant', 'bibliography', 'net worth', 'playing hand', 'ethnic group', 'godparent', 'partnership with', 'member of military unit', 'elected in', 'work period (end)', 'sports discipline competed in', 'era name', 'sport', 'day in year for periodic occurrence', 'officeholder', 'official name', 'applies to jurisdiction', 'residence', 'academic thesis', 'office held by head of the organization', 'owned by', 'student of', 'cause of death', 'religious order', 'studied by', 'significant event', 'language of work or name', 'identifier shared with', 'dedicated to', 'canonization status', 'political alignment', 'ranking', 'has part', 'incarnation of', 'has works in the collection', 'consecrator', 'candidacy in election', 'professorship', 'last words', 'opposite of', 'religious name', 'number of children', 'bowling style', "topic's main template", 'number of matches played/races/starts', 'military branch', 'stepparent', 'second family name in Spanish name', 'honorific suffix', 'present in work', 'category for maps', 'military rank', 'student', 'founded by', 'inception', 'married name', 'capital', 'Roman praenomen', 'candidate', 'languages spoken, written or signed', 'notable work', 'affiliation', 'domain of saint or deity', 'followed by', 'birth name', 'subclass of', 'child', 'conferred by', 'winner', 'official residence', 'copyright status as a creator', 'Fach', 'prize money', 'member of sports team', 'doctoral student', 'country', 'assessment', 'electoral district', 'worshipped by', 'place of detention', 'member of the deme', 'record or record progression', 'feast day', 'appointed by', 'record label', 'mother', 'sexual orientation', 'head coach', 'catchphrase', 'had as last meal', 'item operated', 'occupation', 'place of death', 'organization directed by the office or position', 'supported sports team', 'title', 'member of the crew of', 'country of origin', 'discoverer or inventor', 'lifestyle', 'diocese', 'member of', 'Erdős number', 'instrument', 'unmarried partner', 'uses', 'country for sport', 'head of state', 'country of citizenship', 'patronym or matronym for this name', 'date of baptism in early childhood', 'number of deaths', 'position held', 'award received', 'named after', 'permanent resident of', 'mass', 'spouse', 'time period', 'Roman cognomen', 'personal best', 'time in space', 'conflict', 'educated at', 'penalty', 'inspired by', 'gens', 'league', 'subject has role', 'location', 'has written for', 'filmography', 'short name', 'participant in', 'academic major', 'drafted by', 'iconographic symbol', 'patronym or matronym for this person', 'honorific prefix', 'place of birth', 'copyright representative', 'field of work', 'wears', 'original language of film or TV show', 'astronaut mission', 'handedness', 'follower of', 'date of burial or cremation', 'position played on team/ speciality', 'political ideology', 'family', 'name in native language']


def filter_reln(strrel):
    strrel = re.sub(' +', ' ', strrel)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*~'''
    strrel = re.sub(' +', ' ', strrel)
    strrel = re.sub('_+', '_', strrel)
    for ele in strrel:
        if ele in punc:
            strrel = strrel.replace(ele, "")
    return strrel

def filter_tail(strrel):
    strrel = re.sub(' +', ' ', strrel)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*~'''
    strrel = re.sub(' +', ' ', strrel)
    strrel = re.sub('_+', '_', strrel)
    for ele in strrel:
        if ele in punc:
            strrel = strrel.replace(ele, " ")
    strrel = re.sub(' +', ' ', strrel)
    return strrel

biglist = []

for i in biglist2:
    i = filter_reln(i)
    i = i.lower()
    i = '_'.join(i.split())
    biglist.append(i)



ref_file = open('test-ref.txt')
src_file = open('test-src.txt')
pred_file = open('test-predicted-epoch-5.txt')
jsonl_file = open('test.jsonl')


src_list = src_file.readlines()
pred_list = pred_file.readlines()
ref_list = ref_file.readlines()


counter = 0 


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def extract_ids(reln,pred,tail):
    pred_words = pred.split()

    tail = filter_tail(tail)
    #if tail == 'author' or tail =='writer':
    #    tail = 'author writer'

    if reln.strip() in pred_words:
        indices = [j for j in range(len(pred_words)) if pred_words[j] == reln.strip()]
        
        tail_list = tail.split()
        if len(tail_list) == 1:
            if tail.strip().lower() in pred_words:
                return 1 
        else :
            numk = len(intersection(tail_list, pred_words))
            #print (numk)
            #print (pred,reln,tail)
            if numk > 0 :
                return 1 
        #print ("act: ", reln,tail)
        #print ("pred: ",pred)
        return 0

    else :

        return -1 

lol = 0
gg = 0
sadf = 0
tot = 0
reca = 0
bigtotpred = 0
prec= 0 



bigdict = {
        'te' : {'pr':0,'re':0,'num':0},
        'bn' : {'pr':0,'re':0,'num':0},
        'ta' : {'pr':0,'re':0,'num':0},
        'gu' : {'pr':0,'re':0,'num':0},
        'mr' : {'pr':0,'re':0,'num':0},
        'en':  {'pr':0,'re':0,'num':0},
        'hi' : {'pr':0,'re':0,'num':0},
        'kn' : {'pr':0,'re':0,'num':0},
        }


for i in jsonl_file.readlines() :
    
    i = json.loads(i)
    src = src_list[counter].lower()
    pred = pred_list[counter].lower()
    ref = ref_list[counter].lower()
    pred = filter_reln(pred)


    factlist = i['facts']

    totfacts = len(factlist)
    crpred = 0
    
    inter_list = intersection(pred.split(),biglist)
    totpred = len(set(inter_list))

    templist = []
    for lll in factlist:
        templist.append(lll[0])
   
    if (totpred > totfacts):
        if 'country_of_citizenship' in pred and 'country of citizenship' not in templist:
            totpred -= 0
        if 'occupation' in pred and 'occupation' not in templist:
            totpred -= 0
        else :
            #print ('-'*30)
            #print (i['translated_sentence'])
            #print ("pred: ",pred)
            #print (ref)
            #print (intersection(pred.split(),biglist))
            #print ('-'*30)
            pass
    if totpred == 0:
        totpred = 1
        #print ("pred split", pred.split())
        #print (intersection(pred.split(),biglist))

    bigtotpred += totpred

    lang = i['lang']


    for fc in factlist :
        tot += 1 
        relation = filter_reln(fc[0])
        relation = relation.strip()
        relation = relation.lower()
        relation = '_'.join(relation.split())
        tail = fc[1].lower()
        
        

        number = extract_ids(relation,pred,tail)
        if number == 0 :
            #print ('--'*60)
            lol += 1
            #print (ref)
            #print ("pred: ",pred)
            #print (i['translated_sentence'])
            #print ('--'*60)    

        if number == 1 :
            gg += 1
            crpred += 1

        if number == -1 :
            #print ('--'*60)
            #print (relation)
            #print ("pred: ",pred)
            #print (i['translated_sentence'])
            #print ('--'*60)    
            sadf += 1

    if crpred > totpred:
        crpred = totpred 

    reca += crpred/totfacts
    prec += crpred/totpred
    try:
        bcb += (2* (crpred/totfacts) * (crpred/totpred)) / ((crpred/totfacts) + (crpred/totpred))
    except :
        bcb += 0

    try :
        bigdict[lang]['re'] += crpred/totfacts 
    except: 
        bigdict[lang]['re'] += 0
    bigdict[lang]['pr'] += crpred/totpred
    bigdict[lang]['num'] += 1 
    
    counter+=1 

print (lol)
print (sadf)
print ("num sentences: ",counter)
print ("total predicted: ",bigtotpred)
print ("total references: ",tot)
print ("total correct: ", gg)
print ("total recall: ",gg/tot)
print ("total precision: ",gg/bigtotpred)
print ("avg recall: ",reca/counter)
print ("avg precision: ", prec/counter)
print ("avg f1: ", bcb/counter)


#for i in bigdict:
#    print (i," : ", (200*(bigdict[i]['re']/bigdict[i]['num'])*(bigdict[i]['pr']/bigdict[i]['num']))/((bigdict[i]['re']/bigdict[i]['num'])+(bigdict[i]['pr']/bigdict[i]['num'])))

