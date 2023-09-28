import pandas as pd
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


ref = open('test-ref.txt').readlines()
src = open('test-src.txt').readlines()
pred = open('test-predicted-epoch-7.txt').readlines()


bigdict = {}
mse_error = 0


languages_map = {
    'en': {"label": "English", 'id': 0},
    'hi': {"label": "Hindi", 'id': 1},
    'te': {"label": "Telugu", 'id': 2},
    'bn': {"label": "Bengali", 'id': 3},
    'pa': {"label": "Punjabi", 'id': 4},
    'ur': {"label": "Urdu", 'id': 5},
    'or': {"label": "Odia", 'id': 6},
    'as': {"label": "Assamese", 'id': 7},
    'gu': {"label": "Gujarati", 'id': 8},
    'mr': {"label": "Marathi", 'id': 9},
    'kn': {"label": "Kannada", 'id': 10},
    'ta': {"label": "Tamil", 'id': 11},
    'ml': {"label": "Malayalam", 'id': 12}
}



def get_id_to_lang():
    id_to_lang = {}
    for k, v in languages_map.items():
        id_to_lang[k] = v['label']
    return id_to_lang


id_lang = get_id_to_lang()


def get_rels(istr):
    toret = []
    tmp = istr.split('<R>')
    for j in tmp:
        #print(j)
        tmpj = j.split('<T>')[0].strip().lower()
        toret.append(tmpj)
    toret = [i for i in toret if i ]
    #print (toret)
    return toret



def proc_pad(instr):
    instr = instr.replace('<pad>', '')
    instr = instr.replace(r'/  +/g', ' ');
    return instr


def see_int(lst1,lst2):
    count = 0
    for i in lst1:
        if i in lst2:
            count += 1
    return count



def final_matching(tgt_clusts,pred_clusts):
    '''this function takes two lists of lists as arguments.
    each inner list represents the facts in one sentence.
    The function returns a final labelling score between 0 and 1
    and a kendall tau for ordering '''

    d2_array = []
    sample_count = 0

    for tgt_rels in tgt_clusts:
        sample_count += len(tgt_rels)
        one_row = []
        for pred_rels in pred_clusts:
            number = see_int(tgt_rels,pred_rels)
            one_row.append(number)
        d2_array.append(one_row)

    d2_array = np.array(d2_array)
    row_ind, col_ind = linear_sum_assignment(d2_array,maximize=True)
    fin  = d2_array[row_ind, col_ind].sum()

    tau, p_value = stats.kendalltau(row_ind, col_ind)
    '''
    print (d2_array)
    print (fin)
    print (row_ind,col_ind)
    print ('-'*20)
    '''

    if len(tgt_clusts) == 1 or len(pred_clusts) == 1 :
        tau = 0
        if len(tgt_clusts) == len(pred_clusts):
            tau = 1


    return fin/sample_count, tau



err_count = 0

for lang in ['as','bn','en','hi','gu','kn','ml','mr','or','pa','ta','te']:

    big_number = 0
    overall_corr = 0
    total_count = 0

    for i in range(len(ref)):

        c_src = proc_pad(src[i])
        c_ref = proc_pad(ref[i])
        c_pred = proc_pad(pred[i])
        c_lang = c_src.split()[1]

        if id_lang[lang].lower().strip() != c_lang :
            continue


        prednum = len(c_pred.split('<BR>')) - 1
        tgtnum = len(c_ref.split('<BR>')) -1
        pred_str = c_pred.split('<BR>')[:-1]
        tgt_str = c_ref.split('<BR>')[:-1]


        tgt_clusts = []
        pred_clusts = []

        for tgt_itr in tgt_str:
            ref_rels = get_rels(tgt_itr)
            tgt_clusts.append(ref_rels)
        for pred_itr in pred_str:
            pred_rels = get_rels(pred_itr)
            pred_clusts.append(pred_rels)

        if len(pred_str) == 0 :
            err_count += 1
            continue


        """
        print (c_pred)
        print (pred_str)
        print (tgt_str)
        """

        if (tgtnum > 1 ):
            match_score, corr_score = final_matching(tgt_clusts,pred_clusts)
            big_number += match_score
            overall_corr += corr_score
            total_count += 1

        if prednum >= 10 :
            pass

        mse_error += (tgtnum-prednum)**2
        if (prednum,tgtnum) in bigdict:
            bigdict[(prednum,tgtnum)] += 1
        else :
            bigdict[(prednum,tgtnum)] = 1


    print (lang,big_number/total_count, overall_corr/total_count)

todiv = {}
#print (bigdict)
print (mse_error/len(ref))



for i in range(15):
    for j in range(1,11):
        if (i,j) in bigdict:
            if j in todiv:
                todiv[j] += bigdict[(i,j)]
            else :
                todiv[j] = bigdict[(i,j)]
            #print ((i,j),bigdict[(i,j)],end=' ')



dfd = {}
for i in range(1,11):
    tlist = []
    for j in range(1,11):
        if (j,i) in bigdict:
            #tlist.append(bigdict[j,i]/todiv[i])
            tlist.append(bigdict[j,i])
        else :
            tlist.append(0)
    dfd[i] = tlist

df = pd.DataFrame(dfd,index=range(1,11))
print (df)




true_pos = np.diag(df)
precision = true_pos / np.sum(df, axis=1)
recall = true_pos / np.sum(df, axis=0)


#print (precision,recall)

"""
sns.heatmap(df, cmap="YlGnBu")

plt.xlabel('Actual', fontsize = 11) # x-axis label with fontsize 15
plt.ylabel('Predicted', fontsize = 11) # x-axis label with fontsize 15
plt.show()



"""

print (err_count)
