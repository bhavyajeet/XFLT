import torch
from sacrebleu.metrics import BLEU
import sacrebleu



def get_bl_reward(ref_text, generated_text):
    score = sacrebleu.corpus_bleu([generated_text], [[ref_text]])
    return float(str(score).split()[2])/100

    




def sectitleReward(para, title, titletok, titlemodel, sectitle_device):
    inp = f'<s> {title} {para} </s>'
    emb = titletok(
        inp,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )
    emb = {k: v.to(sectitle_device) for k, v in emb.items()}
    # titlemodel.to(sectitle_device)
    output = titlemodel(**emb)
    output = output.logits
    probs = torch.sigmoid(output)
    argmax = torch.amax(probs, dim=1)
    return argmax.item()

def get_predictions(sentence, tokenizer, model, subwords, device):

    model.to(device)
    tok_sentence = tokenizer(sentence, return_tensors='pt', truncation=True)
    tok_gpu_sentence = {k: v.to(device) for k,v in tok_sentence.items()}

    with torch.no_grad():
        logits = model(**tok_gpu_sentence).logits.argmax(-1)
        predicted_tokens_classes = [model.config.id2label[t.item()] for t in logits[0]]
        predicted_labels = []
        previous_token_id = 0
        word_ids = tok_sentence.word_ids()

        for word_index in range(len(word_ids)):
            if word_ids[word_index] == None:
                previous_token_id = word_ids[word_index]
            elif word_ids[word_index] == previous_token_id:
                previous_token_id = word_ids[word_index]
            else:
                predicted_labels.append( predicted_tokens_classes[ word_index ] )
                previous_token_id = word_ids[word_index]

        current_words = []
        all_words = []
        prev = None
        for i in range(len(word_ids[1:-1])):
            if word_ids[i] != prev:
                word = tokenizer.convert_tokens_to_string(current_words)
                all_words.append(word)
                current_words = []
            current_words.append(subwords[i])
            prev = word_ids[i]
        word = tokenizer.convert_tokens_to_string(current_words)
        all_words.append(word)

    return predicted_labels, all_words

def get_predictions_foriegn(sentence, pipeline):
    ner_results = pipeline(sentence)
    words = [n['word'] for n in ner_results]
    return set(words)

def nerReward(sentence, paragraph, tokenizer, model, device, slang, plang, pipeline):

    if slang:
        ners = get_predictions_foriegn(sentence, pipeline)
    else:
        subwords = tokenizer.tokenize(sentence)
        nertags, allwords = get_predictions(sentence, tokenizer, model, subwords, device)
        ners = set()
        for i in range(len(nertags)):
            if nertags[i] != 'O':
                ners.add(allwords[i])

    if plang:
        pers = get_predictions_foriegn(paragraph, pipeline)
    else:
        parasubwords = tokenizer.tokenize(paragraph)
        pers = set()
        nertags, parawords = get_predictions(paragraph, tokenizer, model, parasubwords, device)
        for i in range(len(nertags)):
            if nertags[i] != 'O':
                pers.add(parawords[i])

    inter = ners & pers

    if len(ners) == 0:
        return 0.5
    elif len(inter) == len(ners):
        return 1
    else:
        return 0

