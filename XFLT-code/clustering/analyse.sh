
langlist=( 'hi' 'en' 'bn' 'ta' 'te' 'mr' 'gu' 'kn' 'ml' 'pa' 'or' 'as' )


for i in "${langlist[@]}"
do
   : 
   python3 analyser.py $i/train.jsonl $i
done

