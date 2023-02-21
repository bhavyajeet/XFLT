lang='ta'
import json 


testfile = open('./test.jsonl')
newfile = open('./lang.jsonl','w',encoding='utf-8')


lines = testfile.readlines()

newlines = []




for i in lines:
    i = json.loads(i)
    if i['lang'] == lang:
        newlines.append(json.dumps(i,ensure_ascii=False)+'\n')


newfile.writelines(newlines)
