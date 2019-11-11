import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizer
import sys
from scipy.spatial.distance import cosine

sys.path.append('../')

from albert.albert_total import get_albert_total
from torch import nn

config_path = '../model/bert/bert_config.json'
model_path = '../model/bert/pytorch_model.bin'
vocab_path = '../model/bert/vocab.txt'


al_config_path = '../model/albert_tiny/bert_config.json'
al_model_path = '../model/albert_tiny/pytorch_model.bin'
al_vocab_path = '../model/albert_tiny/vocab.txt'


class BertTextNet(nn.Module):
    def __init__(self):
        """
        bert模型。
        """
        super(BertTextNet, self).__init__()
        modelConfig = BertConfig.from_pretrained(config_path)
        self.textExtractor = BertModel.from_pretrained(
            model_path, config=modelConfig)
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        return text_embeddings


class BertSeqVec(object):
    def __init__(self, text_net):
        """
        接收一个bert或albert模型，对文本进行向量化。
        :param text_net: bert或albert模型实例。
        """
        self.text_net = text_net
        self.tokenizer = text_net.tokenizer

    def seq2vec(self, text):
        """
        对文本向量化。
        :param text:str，未分词的文本。
        :return:
        """
        text = "[CLS] {} [SEP]".format(text)
        tokens, segments, input_masks = [], [], []

        tokenized_text = self.tokenizer.tokenize(text)  # 用tokenizer对句子分词
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens])  # 最大的句子长度

        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        tokens_tensor = torch.tensor(tokens)
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)
        text_hashCodes = self.text_net(tokens_tensor, segments_tensors,
                                       input_masks_tensors)  # text_hashCodes是bert模型的文本特征
        return text_hashCodes[0].detach().numpy()


class AlbertTextNet(BertTextNet):
    def __init__(self):
        """
        albert 文本模型。
        """
        super(AlbertTextNet, self).__init__()
        config, tokenizer, model = get_albert_total(al_config_path, al_vocab_path, al_model_path)
        self.textExtractor = model
        self.tokenizer = tokenizer

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        return text_embeddings



if __name__ == '__main__':
    texts = ["搭载着中国第36次南极科学考察队队员的“雪龙2”号极地科考破冰船7日与“雪龙”号进行短暂相聚后，\
    离开澳大利亚霍巴特港，将穿越“咆哮西风带”。",
             "这是“雪龙2”号首次穿越西风带，也是其自下水以来所面临的最严峻的一次考验。",
             "我国具有完全自主知识产权的智能动车组昨天（7日）首次在京张高铁线路上参与联调联试，最高测试速度达到350km/h。",
             "京张高铁正线全长174公里，今年底通车后北京到张家口只需50分钟左右。",
             "越南公安部7日晚间发布公告说，越方和英方已确认英国货车惨案39名遇难者均为越南公民。",
             "越南总理阮春福7日代表政府向遇难者亲属致以慰问，并指示越南相关部门和省市继续与英方合作处理相关事宜，早日运回遇难者遗体。",
             "阮春福表示，越南政府强烈谴责人口贩运和组织偷渡等违法行为，呼吁地区和世界各国继续加强合作，打击此类严重犯罪行为，\
             避免悲剧重演，希望案件的调查、起诉和审判工作尽早完成，犯罪分子得到严惩。",
             "美国总统特朗普6日说，他与土耳其总统埃尔多安通电话讨论了叙利亚局势、反恐等问题，他期待埃尔多安下周到访美国。"
             ]
    last_vec = None
    distances = []
    text_net = BertTextNet() # 选择一个文本向量化模型
    seq2vec = BertSeqVec(text_net) #将模型实例给向量化对象。
    for text in texts:
        vec = seq2vec.seq2vec(text) #向量化
        if last_vec is None:
            last_vec = vec
        else:
            dis = cosine(vec, last_vec)
            distances.append(dis)
            last_vec = vec
    print(np.array(distances) - np.mean(np.array(distances)))
    print('done')
