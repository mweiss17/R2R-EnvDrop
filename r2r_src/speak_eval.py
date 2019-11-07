import speaker
from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, read_img_features
from env import R2RBatch
from agent import Seq2SeqAgent
from param import args
from collections import OrderedDict
from eval import Evaluation
from train import setup
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt

setup()
args.angle_feat_size = 128
TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
PLACE365_FEATURES = 'img_features/ResNet-152-places365.tsv'
features = IMAGENET_FEATURES
vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab, encoding_length=80)
feat_dict = read_img_features(features)

train_env = R2RBatch(feat_dict, batch_size=64, splits=['train'], tokenizer=tok)
log_dir = "snap/speaker/state_dict/best_val_seen_bleu"
val_env_names = ['val_unseen', 'val_seen']
featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

val_envs = OrderedDict(
    ((split,
      (R2RBatch(feat_dict, batch_size=args.batchSize, splits=[split], tokenizer=tok),
       Evaluation([split], featurized_scans, tok))
      )
     for split in val_env_names
     )
)

listner = Seq2SeqAgent(train_env, "", tok, 35)
speaker = speaker.Speaker(train_env, listner, tok)
speaker.load(log_dir)
speaker.env = train_env
results = {}
for env_name, (env, evaluator) in val_envs.items():
    print("............ Evaluating %s ............." % env_name)
    speaker.env = env
    path2inst, loss, word_accu, sent_accu = speaker.valid()

    r = defaultdict(dict)
    for path_id in path2inst.keys():
        # internal_bleu = evaluator.compute_internal_bleu_score(path_id)
        # if internal_bleu == 1.0:
        #     import pdb;
        #     pdb.set_trace()

        internal_bleu = evaluator.compute_internal_bleu_score(path_id)
        external_bleu = evaluator.bleu_score({path_id: path2inst[path_id]})[0]
        p = {"inference": tok.decode_sentence(path2inst[path_id]), "gt": evaluator.gt[str(path_id)]['instructions'], "internal_bleu": internal_bleu, "external_bleu": external_bleu}
        r[path_id] = p
    results[env_name] = r

for env_name in ["val_unseen", "val_seen"]:
    mean_internal_bleu = np.mean([v['internal_bleu'] for k, v in results[env_name].items()])
    mean_external_bleu = np.mean([v['external_bleu'] for k, v in results[env_name].items()])

    plt.hist([v['external_bleu'] for k, v in results[env_name].items()], np.linspace(0, 1, 50), label="External BLEU Score", alpha=0.3)
    plt.hist([v['internal_bleu'] for k, v in results[env_name].items()], np.linspace(0, 1, 50), label="Internal BLEU Score", alpha=0.3)
    plt.title(f"BLEU Score on {env_name}")
    plt.xlabel("BLEU Score Bins")
    plt.legend(loc="upper right")
    plt.ylabel('# Instructions in Bin')
    plt.savefig(f"{env_name}.png")
    plt.cla()
    plt.clf()
import pdb; pdb.set_trace()
pass
