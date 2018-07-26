import gensim
import numpy as np
import jieba
import config

USERDICT = config.USERDICT
STOPWORDS = config.STOPWORDS
W2VMODEL = config.W2VMODEL


jieba.load_userdict(USERDICT)

with open(STOPWORDS, 'r', encoding='utf-8') as f:
    stopwords = set([tag.strip() for tag in f.readlines()])

model = gensim.models.Word2Vec.load(W2VMODEL)
try:
    model_vocab = model.vocab
except Exception:
    model_vocab = model.wv.vocab


def cut_words(sentence):
    words = " ".join([word.strip() for word in jieba.cut(sentence.strip()) if word not in stopwords])
    return words


def sentence2vector(sencence, n_step):
    words = cut_words(sencence)
    sencence_words = words.split()
    if len(sencence_words) < n_step:
        sencence_words += ['<UNK>'] * (n_step - len(sencence_words))
    else:
        sencence_words = sencence_words[:n_step]
    words_list = []
    for word in sencence_words:
        if word in model_vocab:
            vec = model[word]
        else:
            vec = [0.0] * 256
        words_list.append(vec)
    return np.array(words_list, dtype=np.float32)




