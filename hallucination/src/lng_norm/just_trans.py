from googletrans import Translator
import matplotlib.pyplot as plt 
import random
import pandas as pd

df = pd.read_csv('./new.csv',header=0)

biglist = []
leftlist = []

count = 0 


translator = Translator()


for index, row in df.iterrows():
    #print(row['filtered'], row['label'])
    if row['language'].lower() == 'chinese' and row['label'] > 2.4 and row['label'] < 4.2:
        biglist.append(row)
    elif row['language'].lower() == 'chinese' :
        leftlist.append(row)












print (len(biglist))
other  = random.choices(leftlist, k=200)

fin = other + biglist

fin = pd.DataFrame(fin)
#print (fin)


newdf = []


for i,row in fin.iterrows():
    print (i)
    translation = translator.translate(row['filtered'],dest='ko')
    bigdict = {}
    bigdict['text'] = translation.text
    bigdict['label'] = row['label']
    bigdict['language'] = 'Korean'
    bigdict['filtered'] = translation.text
    newdf.append(bigdict)



plt.hist(fin['label'], bins=20, color='#86bf91', zorder=2, rwidth=0.9)


bigdf = pd.DataFrame(newdf)
bigdf.to_csv('./korean_sample.csv',index=False)

#print (df)



