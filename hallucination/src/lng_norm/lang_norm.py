import sys
from tqdm import tqdm 
from googletrans import Translator


lang = sys.argv[1]

nlist = open('nouns.txt').readlines()

nlist = [x.strip() for x in nlist]

translator = Translator()

anlist = []
bigstr = ''


for x in tqdm(nlist) :
    translation = translator.translate(x,dest=lang)
    anlist.append(translation.text)
    bigstr += translation.text +'\n'


newfile = open(lang+'_nouns.txt','w')

newfile.write(bigstr)


