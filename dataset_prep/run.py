# Prepare sentence list for train dataset

[ 'hi','en','bn','ta','te','mr','gu','kn','ml','pa','or','as' ]

python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/en/train.jsonl en train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/hi/train.jsonl hi train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/bn/train.jsonl bn train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/ta/train.jsonl ta train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/te/train.jsonl te train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/mr/train.jsonl mr train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/gu/train.jsonl gu train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/kn/train.jsonl kn train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/ml/train.jsonl ml train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/pa/train.jsonl pa train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/or/train.jsonl or train 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/as/train.jsonl as train 


# Prepare sentence list for val dataset
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/en/val.jsonl en val
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/hi/val.jsonl hi val 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/bn/val.jsonl bn val 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/ta/val.jsonl ta val 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/te/val.jsonl te val 
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/mr/val.jsonl mr val  
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/gu/val.jsonl gu val  
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/kn/val.jsonl kn val  
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/ml/val.jsonl ml val  
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/pa/val.jsonl pa val  
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/or/val.jsonl or val  
python analyser.py /Users/rahulmehta/Desktop/MultiSent/datasets/datasets_v2.7/as/val.jsonl as val 


# Prepare dataset for all languages
python converter.py train
python converter.py val 