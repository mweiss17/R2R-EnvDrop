import nltk
import pickle
from collections import Counter

tagged_instructions = []
all_results = pickle.load(open("results.pkl", 'rb'))
for env_name, results in all_results.items():
    for path_id, result in results.items():
        inf = (result['inference'])
        toks = nltk.word_tokenize(inf)
        tagged_instructions.append(nltk.pos_tag(toks))

nouns = Counter()
for tagged_instruction in tagged_instructions:
    for word, tag in tagged_instruction:
        if tag == "NN":
            nouns[word] += 1

infs = [x['inference'] for x in results.values()]
gts = [x['gt'] for x in results.values()]
last_sentences_infs = [inf.split(".")[-2] for inf in infs]
last_sentences_gts = [gt[0].split(".")[-2] for gt in gts]

cleaned = []
for sentence in last_sentences_gts:
    if "and" in sentence or "right" in sentence or "left" in sentence:
        continue
    cleaned.append(sentence)
import pdb; pdb.set_trace()
print(cleaned)
#
# print(nouns.most_common(100))