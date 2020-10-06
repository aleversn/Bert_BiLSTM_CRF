# %%
from ner.trainer import NERTrainer

# %%
trainer = NERTrainer(10, [0, 1, 2, 3],
                     bert_config_file_name='./model/chinese_wwm_ext/bert_config.json',
                     pretrained_file_name='./model/chinese_wwm_ext/pytorch_model.bin',
                     hidden_dim=150,
                     train_file_name='./data/news/train.txt',
                     vocab_file_name='./model/chinese_wwm_ext/vocab.txt',
                     tags_file_name='./data/news_tags_list.txt',
                     eval_file_name='./data/news/test.txt',
                     batch_size=250,
                     eval_batch_size=64)

for i in trainer():
    a = i

# %%
from ner.predicter import NERPredict

# %%
predict = NERPredict(True,
                     bert_config_file_name='./model/chinese_wwm_ext/bert_config.json',
                     vocab_file_name='./model/chinese_wwm_ext/vocab.txt',
                     tags_file_name='./data/news_tags_list.txt',
                     bert_model_path='./save_model/bert/b38f6126_bert.pth',
                     lstm_crf_model_path='./save_model/lstm_crf/b38f6126_lstm_crf.pth',
                     hidden_dim=150)

# %%
print(predict(["坐落于福州的福州大学ACM研究生团队, 在帅气幽默的傅仰耿老师带领下, 正在紧张刺激的开发一套全新的神秘系统。","在福州大学的后山, 驻扎着福大后山协会, 会长是陈学勤同志。"])[2:])


# %%
