from sacrebleu.metrics import BLEU

lang  = 'hi'

filen = '/scratch/model_outputs/towork/results-all-euclidean/'+lang+'-coverage.jsonl'

out_ref = open('./ref-file').readlines()
out_og_gen = open('./og-file').readlines()
out_new_gen = open('./new-file').readlines()

#bleu = BLEU(tokenize='char')
bleu = BLEU()

result = bleu.corpus_score( out_og_gen,[out_ref])
print (result)


result = bleu.corpus_score(out_new_gen,[out_ref])
print (result)


    

