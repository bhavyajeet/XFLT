import os 
import json
import ast
import csv 
from collections import Counter



if __name__ == "__main__":
    output_path = f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/scores/' 
    #output_path = f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/ref/scores/' 

    with open(output_path + 'coverage-scores.csv', 'a', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames = ["distance", "threshold", "lang","coverage","avg_coverage","total_correct","total_sentlen"])
        writer.writeheader()

    DISTANCES = ['euclidean','cosine']
    THRESHOLDS = [0.20,0.40,0.65,0.70,0.75]

    for DISTANCE in DISTANCES:
        root = f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/results-all-' + DISTANCE + '/' # If ref
        #root = f'/Users/rahulmehta/Desktop/MultiSent/hallucination/datasets/results-all-' + DISTANCE + '/'
        for THRESHOLD in THRESHOLDS:
            for subdir, dirs, files in os.walk(root):

                #for file in files:
                for file in files:#['ka-coverage.jsonl']:
                    
                    #print(file)

                    lang = file[0:2]

                    total_correct = 0
                    total_sentlen = 0
                    total_coverage = 0

                    with open(root + file,'r') as f:
                        line_cnt = 0
                        for line in f:
                            line_cnt+=1
                            data = json.loads(line)
                            print(data)

                            facts = data["facts"]
                            scores =  data["scores"]
                            #print(scores)
                            sentence =data["sentence"]

                            # Denominator 
                            sentence_len = len(data['sentence'].split(' '))
                            

                            # Numerator
                            correct =0
                            cor = []
                            print(scores)
                            for item in scores:
                                #print(tuple(item))
                                if item[2] < THRESHOLD and item[1] in facts:
                                    #correct+=1
                                    #print(item[1],correct)
                                    cor.append(item[1])
                            cnt = Counter(cor).keys()
                            print(cnt)
                            correct += len(cnt)       

                            coverage = correct/sentence_len
                            total_correct +=correct 
                            total_sentlen += sentence_len
                            total_coverage += coverage
                            print(correct,sentence_len,coverage)
                    
                        
                    print("file {} Coverage {}".format(lang,total_correct/total_sentlen))
                    print("file {} Average Coverage {}".format(lang,total_coverage/line_cnt))


                    # with open(output_path +'/scores.txt','a') as s:
                    #     s.write("distance %s " % DISTANCE)
                    #     s.write("threshold %s " % THRESHOLD)
                    #     s.write("language %s " % lang)
                    #     s.write("coverage {0} ".format(total_correct/total_sentlen))
                    #     s.write("average sent coverage {0} ".format(total_coverage/line_cnt))
                    #     s.write("correct {0} total words {1}".format(total_correct,total_sentlen))
                    #     s.write("\n")


                    
                    with open(output_path + 'coverage-scores.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([DISTANCE,THRESHOLD,lang,total_correct/total_sentlen,total_coverage/line_cnt,total_correct,total_sentlen])



