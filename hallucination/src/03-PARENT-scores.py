import os 
import json
import ast

DISTANCE = 'cosine'
THRESHOLD = 0.65

if __name__ == "__main__":
    root = f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/results-all-' + DISTANCE + '/'
    output_path = f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/scores' 

    for subdir, dirs, files in os.walk(root):
        for file in files:
            print(file)

            lang = file[0:2]

            total_correct = 0
            total_sentlen = 0
            with open(root + file,'r') as f:
                for line in f:
                    
                    data = json.loads(line)
                    #print(data)

                    facts = data["facts"]
                    scores =  data["scores"]
                    #print(scores)
                    sentence =data["sentence"]

                    # Denominator 
                    sentence_len = len(data['sentence'].split(' '))
                    

                    # Numerator
                    correct =0
                    for item in scores:
                        #print(tuple(item))
                        if item[2] > THRESHOLD and item[1] in facts:
                            correct+=1
                            print(item[1],correct)

                    coverage = correct/sentence_len
                    total_correct +=correct 
                    total_sentlen += sentence_len
                    print(correct,sentence_len,coverage)
            
            print("file {} Coverage {}".format(lang,total_correct/total_sentlen))


            with open(output_path +'/scores.txt','a') as s:
                s.write("language %s " % lang)
                s.write("coverage {0} ".format(total_correct*100/total_sentlen))
                s.write("correct {0} total words {1}".format(total_correct,total_sentlen))
                s.write("\n")




