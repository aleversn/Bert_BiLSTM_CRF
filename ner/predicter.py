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
from ICCSupervised.ICCSupervised import IPredict
from ner.dataloader import CCDataLoader, BERTDataManager
from ner.analysis import CCAnalysis
from ner.model import BertNER


class NERPredict(IPredict):

    def __init__(self, use_gpu,
                 bert_config_file_name,
                 vocab_file_name,
                 tags_file_name,
                 bert_model_path,
                 lstm_crf_model_path,
                 hidden_dim):
        self.use_gpu = use_gpu
        self.data_manager_init(vocab_file_name, tags_file_name)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file_name)
        self.model_init(hidden_dim, bert_config_file_name,
                        bert_model_path, lstm_crf_model_path)

    def data_manager_init(self, vocab_file_name, tags_file_name):
        tags_list = BERTDataManager.ReadTagsList(tags_file_name)
        tags_list = [tags_list]
        self.dm = BERTDataManager(tags_list=tags_list,
                                  vocab_file_name=vocab_file_name)

    def model_init(self, hidden_dim, bert_config_file_name, bert_model_path, lstm_crf_model_path):
        config = BertConfig.from_json_file(bert_config_file_name)

        self.model = BertModel(config)

        bert_dict = torch.load(bert_model_path).module.state_dict()

        self.model.load_state_dict(bert_dict)
        self.birnncrf = torch.load(lstm_crf_model_path)

        self.model.eval()
        self.birnncrf.eval()

    def data_process(self, sentences):
        result = []
        pad_tag = '[PAD]'
        if type(sentences) == str:
            sentences = [sentences]
        max_len = 0
        for sentence in sentences:
            encode = self.tokenizer.encode(sentence, add_special_tokens=True)
            result.append(encode)
            if max_len < len(encode):
                max_len = len(encode)

        for i, sentence in enumerate(result):
            remain = max_len - len(sentence)
            for _ in range(remain):
                result[i].append(self.dm.wordToIdx(pad_tag))
        return torch.tensor(result)

    def pred(self, sentences):
        sentences = self.data_process(sentences)

        if torch.cuda.is_available():
            self.model.cuda()
            self.birnncrf.cuda()
            sentences = sentences.cuda()

        outputs = self.model(input_ids=sentences,
                             attention_mask=sentences.gt(0))
        hidden_states = outputs[0]
        scores, tags = self.birnncrf(hidden_states, sentences.gt(0))
        final_tags = []
        decode_sentences = []

        for item in tags:
            final_tags.append([self.dm.idx_to_tag[tag] for tag in item])
        
        for item in sentences.tolist():
            decode_sentences.append(self.tokenizer.decode(item))

        return (scores, tags, final_tags, decode_sentences)
    
    def __call__(self, sentences):
        return self.pred(sentences)
