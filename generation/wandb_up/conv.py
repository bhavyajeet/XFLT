import torch 
import sys 


pt_name = sys.argv[1]

new = {}

person = {}

lol = torch.load(pt_name)

for i in lol:
    k=i[7:]
    new[k] = lol[i]
    
person['state_dict'] = new


torch.save(person, sys.argv[2] )

