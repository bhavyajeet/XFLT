from transformers import AutoTokenizer, AutoModel
from functools import lru_cache

#import functorch
import regex as re 
import torch 
import time 
import torch.nn.functional as F

sample_in = "generate english : <H> saiful islam joarder <R> country_of_citizenship <T> bangladesh <R> religion <T> islam <R> military_rank <T> lieutenant colonel <R> occupation <T> military personnel"
sample_in = "generate english : charlie townsend date_of_death 17 october 1958 date_of_birth 07 november 1876 occupation cricketer"

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model = AutoModel.from_pretrained("google/muril-base-cased", output_hidden_states=True)


def get_embedding(tokens):
    #print(tokens)
    with torch.no_grad():
        tokenized_facts = tokenizer(tokens, padding=False, truncation=False, return_tensors="pt")
        #print(tokenizer.convert_ids_to_tokens(tokenized_facts['input_ids'][0]))
        states = model(**tokenized_facts).hidden_states
        output = torch.stack([states[i] for i in range(len(states))])
        output = output.squeeze()
        final_hidden_state = torch.mean(output[-4:-1, :, ...], dim=0)
        #print(final_hidden_state.shape)
        return final_hidden_state
        #return embeddings['last_hidden_state'] #tokenized_facts['attention_mask']
        #return torch.mean(embeddings['last_hidden_state'], dim=1)

def get_similarity(token, input_sent, metric):
    words = input_sent.split(" ")
    language = words[1]
    facts = re.split(r'<.>', " ".join(words[3:]))
    facts = [re.sub(r'_', ' ', f) for i in range(len(facts))  for f in facts[i].split()]
    fact_embedding = get_embedding(" ".join(facts))[1:-1]
    #print(fact_embedding.shape)
    #fact_embedding = fact_embedding.reshape(fact_embedding.shape[0]*fact_embedding.shape[1], -1)
    #print(fact_embedding.shape)
    #print (token)
    token_embedding = get_embedding([token,])[1:-1]
    #print(token_embedding.shape, fact_embedding.shape)
    #token_embedding = token_embedding.repeat(fact_embedding.shape[0],1)
    #print(token_embedding.shape, fact_embedding.shape)
    sims = []
    for i in range(token_embedding.shape[0]):
        token_sims = []
        for j in range(fact_embedding.shape[0]):
            token_sims.append(metric(token_embedding[i:i+1], fact_embedding[j:j+1]))
        sims.append(token_sims)

    sims = torch.as_tensor(sims)
    if len(sims) == 0:
        return torch.as_tensor([0.1], device=sims.device)
    return sims.max(dim=1).values
    #return functorch.vmap(lambda x, y: metric(x, y))(token_embedding, fact_embedding)

if __name__ == '__main__':
    #st = time.time()
    print(get_similarity('Anthony', sample_in, F.cosine_similarity))
    print(get_similarity('Townsend', sample_in, F.cosine_similarity))
    print(get_similarity('Town', sample_in, F.cosine_similarity))
    print(get_similarity('townsend', sample_in, F.cosine_similarity))
    #en = time.time()
    #print(en-st)
    #print(get_similarity('islamicism', sample_in, F.pairwise_distance))
    #print(time.time()-en)
