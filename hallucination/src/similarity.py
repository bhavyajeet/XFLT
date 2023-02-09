from transformers import AutoTokenizer, AutoModel
from functools import lru_cache

import functorch
import regex as re 
import torch 
import time 
import torch.nn.functional as F

sample_in = "generate english : <H> saiful islam joarder <R> country_of_citizenship <T> bangladesh <R> military_branch <T> bangladesh army <R> military_rank <T> lieutenant colonel <R> occupation <T> military personnel"

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased", cache_dir='/scratch')
model = AutoModel.from_pretrained("google/muril-base-cased")


@lru_cache
def get_embedding(tokens):
    print(tokens)
    with torch.no_grad():
        tokenized_facts = tokenizer(tokens, padding=False, truncation=False, return_tensors="pt")
        print(tokenizer.convert_ids_to_tokens(tokenized_facts['input_ids'][0]))
        embeddings = model(**tokenized_facts)
        return embeddings['last_hidden_state'] #tokenized_facts['attention_mask']
        #return torch.mean(embeddings['last_hidden_state'], dim=1)

def get_similarity(token, input_sent, metric):
    words = input_sent.split(" ")
    language = words[1]
    facts = re.split(r'<.>', " ".join(words[3:]))
    facts = [re.sub(r'_', ' ', f) for i in range(len(facts))  for f in facts[i].split()]
    fact_embedding = get_embedding(" ".join(facts))[0][1:-1]
    #print(fact_embedding.shape)
    #fact_embedding = fact_embedding.reshape(fact_embedding.shape[0]*fact_embedding.shape[1], -1)
    #print(fact_embedding.shape)
    token_embedding = get_embedding(tuple([token,]))[0][1:-1]
    print(token_embedding.shape, fact_embedding.shape)
    token_embedding = token_embedding.repeat(fact_embedding.shape[0], 1, 1)
    #print(token_embedding.shape)
    return functorch.vmap(lambda x, y: metric(x, y))(token_embedding, fact_embedding)


st = time.time()
print(get_similarity('islam', sample_in, F.cosine_similarity))
en = time.time()
# print(en-st)
# print(get_similarity('bangladesh', sample_in, "cosine"))
# print(time.time()-en)


