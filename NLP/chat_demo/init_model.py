import qa_model
import config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model1 = qa_model.QaModel(checkpoint_dir=config.checkpoint_dir_shot, tree_name=config.tree_name.get('shot'),
                          table_name=config.table_name.get('shot'), query_steps=12)
model2 = qa_model.QaModel(checkpoint_dir=config.checkpoint_dir_long, tree_name=config.tree_name.get('long'),
                          table_name=config.table_name.get('long'), query_steps=100)
