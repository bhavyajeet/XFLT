## Structure 
The evaluation script requires ```test-src.txt```, ```test-ref.txt``` and ```test-predicted.txt``` and ```test.jsonl```. ```sample_path``` contains examples of the required files. Note that the same nomenclature must be followed. 

## Sample script 
python main.py --root_file sample_path --test_path sample_path/test.jsonl --name sample --single 1 --exclusion 0 --parent 0 --threshold none
