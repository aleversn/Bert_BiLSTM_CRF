import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from transformers import BertConfig, BertTokenizer, BertModel
from ner.crf import CRF
from ner.model_with_bert import BiRnnCrf
from tqdm import tqdm
from ICCSupervised.ICCSupervised import ITrainer
from ner.dataloader import CCDataLoader
from ner.analysis import CCAnalysis
from ner.model import BertNER


'''
NER训练器
'''
class NERTrainer(ITrainer):
    
    '''
    构造函数, 初始化训练器
    num_epochs: 训练迭代次数
    num_gpus: GPU应用数, 如: [0, 1, 2, 3]
    bert_config_file_name: Bert模型配置文件路径
    pretrained_file_name: Bert预训练模型加载路径
    hidden_dim: CRF隐藏层
    train_file_name: 训练集文件路径
    vocab_file_name: 单词表文件路径
    tags_file_name: Tag表文件路径
    eval_file_name: 测试集文件路径
    word_tag_split: 文字Tag分隔符
    pattern: 断句符
    padding_length: 填充裁剪长度
    batch_size: 训练集batch_size
    eval_batch_size: 测试集batch_size, 缺省时等于训练集batch_size
    '''
    def __init__(self, num_epochs, num_gpus,
                 bert_config_file_name,
                 pretrained_file_name,
                 hidden_dim,
                 train_file_name,
                 vocab_file_name,
                 tags_file_name,
                 eval_file_name=None,
                 word_tag_split=' ',
                 pattern='， O',
                 padding_length=50,
                 batch_size=32,
                 eval_batch_size=None):

        self.eval_data = True if eval_file_name != None else False
        self.num_epochs = num_epochs
        self.num_gpus = num_gpus
        self.dataloader_init(train_file_name, vocab_file_name, tags_file_name, word_tag_split,
                             pattern, padding_length, batch_size, eval_file_name, eval_batch_size)
        self.vocab_size = len(self.dm.word_to_idx)
        self.tagset_size = len(self.dm.tag_to_idx)
        self.model_init(bert_config_file_name,
                        pretrained_file_name, hidden_dim)

    def model_init(self, bert_config_file_name, pretrained_file_name, hidden_dim):
        self.bert_ner = BertNER(bert_config_file_name, pretrained_file_name,
                                self.vocab_size, self.tagset_size, hidden_dim)
        self.model, self.birnncrf = self.bert_ner()

    def dataloader_init(self, train_file_name, vocab_file_name, tags_file_name, word_tag_split, pattern, padding_length, batch_size, eval_file_name, eval_batch_size):
        self.dataloader = CCDataLoader(
            train_file_name, vocab_file_name, tags_file_name, word_tag_split, eval_file_name, pattern, padding_length, batch_size, eval_batch_size)
        self.mydata, self.dataiter, self.dm = self.dataloader()[0]
        if self.eval_data:
            self.mydata_eval, self.dataiter_eval = self.dataloader()[1]
        self.analysis = CCAnalysis(self.dm)

    def train(self):
        alpha = 1e-10

        optimizer = optim.Adam([
            {'params': self.model.parameters(), 'lr': 2e-5},
            {'params': self.birnncrf.parameters(), 'lr': 1e-3}
        ], lr=1e-5, weight_decay=0.)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.num_gpus)
            self.birnncrf.cuda()

        self.model.to(device)
        self.birnncrf.to(device)

        current_epoch = 0
        for epoch in range(self.num_epochs):
            current_epoch += 1
            train_count = 0
            train_pred_num = 0
            train_gold_num = 0
            train_correct_num = 0
            train_loss = 0
            train_iter = tqdm(self.dataiter)
            self.model.train()
            self.birnncrf.train()
            for sentence, tags in train_iter:
                if torch.cuda.is_available():
                    sentence = Variable(sentence.cuda())
                    tags = Variable(tags.cuda())
                else:
                    sentence = Variable(sentence)
                    tags = Variable(tags)

                outputs = self.model(input_ids=sentence,
                                     attention_mask=sentence.gt(0))
                hidden_states = outputs[0]
                loss = self.birnncrf.loss(hidden_states, sentence.gt(0), tags)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                self.model.zero_grad()
                self.birnncrf.zero_grad()

                train_loss += loss.data.item()
                train_count += 1

                pred = self.birnncrf(hidden_states, sentence.gt(0))[1]

                t1, t2 = self.analysis.getPrecision(tags, pred)
                train_pred_num += t1
                train_correct_num += t2
                train_gold_num += self.analysis.getRecall(tags)

                train_acc = train_correct_num / train_pred_num if train_correct_num != 0 else 0
                train_recall = train_correct_num / train_gold_num if train_correct_num != 0 else 0

                train_iter.set_description(
                    'Epoch: {}/{} Train'.format(current_epoch, self.num_epochs))
                train_iter.set_postfix(train_loss=train_loss / train_count, train_acc=train_acc,
                                       train_recall=train_recall, F1=(2 * train_acc * train_recall) / (train_acc + train_recall + alpha))
                self.analysis.append_train_record({
                    'loss': loss.data.item(),
                    'f1': (2 * train_acc * train_recall) / (train_acc + train_recall + alpha),
                    'acc': train_acc,
                    'recall': train_recall
                })
            
            self.save_model()
            if self.eval_data:
                self.eval()
        
        self.analysis.save_csv()
        yield (current_epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record)

    def eval(self):
        alpha = 1e-10

        test_count = 0
        test_pred_num = 0
        test_gold_num = 0
        test_correct_num = 0
        eval_loss = 0
        test_iter = tqdm(self.dataiter_eval)
        self.model.eval()
        self.birnncrf.eval()
        for sentence, tags in test_iter:
            if torch.cuda.is_available():
                sentence = Variable(sentence.cuda())
                tags = Variable(tags.cuda())
            else:
                sentence = Variable(sentence)
                tags = Variable(tags)

            outputs = self.model(input_ids=sentence,
                                 attention_mask=sentence.gt(0))
            hidden_states = outputs[0]
            loss = self.birnncrf.loss(hidden_states, sentence.gt(0), tags)
            loss = loss.mean()

            eval_loss += loss.data.item()
            test_count += 1

            pred = self.birnncrf(hidden_states, sentence.gt(0))[1]

            t1, t2 = self.analysis.getPrecision(tags, pred)
            test_pred_num += t1
            test_correct_num += t2
            test_gold_num += self.analysis.getRecall(tags)

            test_acc = test_correct_num / test_pred_num if test_correct_num != 0 else 0
            test_recall = test_correct_num / test_gold_num if test_correct_num != 0 else 0

            test_iter.set_description('Eval Result')
            test_iter.set_postfix(eval_loss=eval_loss / test_count, eval_acc=test_acc, eval_recall=test_recall,
                                  F1=(2 * test_acc * test_recall) / (test_acc + test_recall + alpha))
        self.analysis.append_eval_record({
            'loss': loss.data.item(),
            'f1': (2 * test_acc * test_recall) / (test_acc + test_recall + alpha),
            'acc': test_acc,
            'recall': test_recall
        })

    def save_model(self):
        uid = str(uuid.uuid1()).split('-')[0]
        if not os.path.exists('./save_model/bert'):
            os.makedirs('./save_model/bert')
        if not os.path.exists('./save_model/lstm_crf'):
            os.makedirs('./save_model/lstm_crf')
        torch.save(self.model, './save_model/bert/{}_bert.pth'.format(uid))
        torch.save(self.birnncrf,
                   './save_model/lstm_crf/{}_lstm_crf.pth'.format(uid))
        self.analysis.append_model_record(uid)
    
    def __call__(self):
        for r in self.train():
            yield r
