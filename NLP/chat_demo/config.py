import tensorflow as tf

USERDICT = './data/all.dic'
STOPWORDS = './data/stopwords'
W2VMODEL = './data/w2v/w2v.model'
DATABASE = './data/qa.db'

checkpoint_dir_shot = './data/checkpoints_shot'
checkpoint_dir_long = './data/checkpoints_long'

tree_name = {
    'shot': './data/shot.ann',
    'long': '/home/data/fyl/chat_model2/qa/nwxanan.ann'
}

table_name = {
    'shot': 'ananyisheng_query_shot',
    'long': 'wx_chat'
}
