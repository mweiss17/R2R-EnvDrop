import pickle
from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt

TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab, encoding_length=80)

inf_count = Counter()
ref_count = Counter()
# for item in data:
#     for instr in item['instructions']:
#         count.update(Tokenizer.split_sentence(instr))
# vocab = list(start_vocab)
# for word, num in count.most_common():
#     if num >= min_count:
#         vocab.append(word)
#     else:
#         break

all_results = pickle.load(open("results.pkl", 'rb'))
for env_name, results in all_results.items():
    print(env_name)
    for path_id, result in results.items():
        print(result)
        inf = tok.split_sentence(result['inference'])
        inf_count.update(inf)
        ref = np.random.choice([tok.split_sentence(x) for x in result['gt']])
        ref_count.update(ref)

infs = sorted(inf_count.values(), reverse=True)
refs = sorted(ref_count.values(), reverse=True)

plt.plot(infs, label="Inferred Language")
plt.plot(refs, label="Train Language")
plt.title(f"Distribution of Vocabulary")
plt.xlabel("Words")
plt.legend(loc="upper right")
plt.ylabel('Amount of Usage')
plt.savefig("vocab_dist.png")
plt.cla()
plt.clf()
