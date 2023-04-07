import re
from io import open
import jieba
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from bert_score import BERTScorer

scorer = BERTScorer(model_type="/home/zxy/code/SportsSum/model", num_layers=8)

train_path = r"/home/zxy/code/SportsSum/my_rewriter/data/train.txt"
val_path = r"/home/zxy/code/SportsSum/my_rewriter/data/val.txt"
test_path = r"/home/zxy/code/SportsSum/my_rewriter/data/test.txt"

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 初始化前置符号
init_word = ['_pad_', '_unk_', '_sos_', '_eos_']
content_set = set()
title_set = set()


# 分词
def cut(text):
    # jieba.load_userdict('/home/zxy/code/SportsSum/my_rewriter/data/linesup_group.txt')
    return [i for i in jieba.cut(text)]


# 打开文件
with open(train_path, 'r', encoding='utf8') as f:
    for line in tqdm.tqdm(f):
        content, title = line.split("&")
        title_set.update(cut(title))
        content_set.update(cut(content))

with open(val_path, 'r', encoding='utf8') as f:
    for line in tqdm.tqdm(f):
        content, title = line.split("&")
        title_set.update(cut(title))
        content_set.update(cut(content))

with open(test_path, 'r', encoding='utf8') as f:
    for line in tqdm.tqdm(f):
        content, title = line.split("&")
        title_set.update(cut(title))
        content_set.update(cut(content))

# 词典
content_list = list(content_set)
title_list = list(title_set)


# 保存
def save_dict(dict_list, file):
    with open(file, 'w') as f:
        f.write('\n'.join(dict_list))


# 过滤无用词
def filter_useless(dict_list, name):
    def is_zh(s):
        zhmodel = re.compile(u'[\u4e00-\u9fa5]|[0-9]+|[a-zA-Z]')
        res = zhmodel.search(s)
        if res:
            return True

    tmp = []
    for i in dict_list:
        if is_zh(i):
            tmp.append(i)
    save_dict(init_word + tmp, name)


filter_useless(content_list, 'content_dict.txt')
filter_useless(title_list, 'title_dict.txt')


# 统计标题及正文 平均长度
def count_ave_len(file_name):
    len_title = []
    len_content = []
    with open(file_name, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            content, title = line.split("&")
            len_title.append(len(title))
            len_content.append(len(content))
    print("\nave_len is", np.mean(np.array(len_title)), np.mean(np.array(len_content)))


count_ave_len(train_path)
count_ave_len(val_path)
count_ave_len(test_path)


class Vocab:
    def __init__(self, filepath):
        self.filepath = filepath
        self.dict, self.dict_reverse = self.load_dict()

    def load_dict(self):
        tmp_dict = {}
        tmp_dict_reverse = {}
        with open(self.filepath) as f:
            for index, i in enumerate(f):
                tmp_dict[i.replace('\n', '')] = int(index)
                tmp_dict_reverse[str(index)] = i.replace('\n', '')
        return tmp_dict, tmp_dict_reverse

    def get_dict_dim(self):
        return len(self.dict)

    def content2id(self, txt_seq, max_que_len=70):
        # jieba.load_userdict('/home/zxy/code/SportsSum/my_rewriter/data/linesup_group.txt')
        que_line = jieba.lcut(txt_seq)
        que_dict = self.dict
        que_len = len(que_line)
        que_list = list()
        for i in range(len(que_line)):
            if que_line[i] in que_dict.keys():
                que_list.append(que_dict[que_line[i]])
            # else:
            #     que_list.append(que_dict['_unk_'])
        if len(que_list) < max_que_len:
            que_list += [que_dict['_pad_'] for _ in range(max_que_len - len(que_list))]
        else:
            que_list = que_list[:max_que_len]
        return que_list

    def title2id(self, title_seq, max_que_len=70):
        # jieba.load_userdict('/home/zxy/code/SportsSum/my_rewriter/data/linesup_group.txt')
        ans_line = jieba.lcut(title_seq)
        ans_dict = self.dict
        ans_list = list()
        for i in range(len(ans_line)):
            if ans_line[i] in ans_dict.keys():
                ans_list.append(ans_dict[ans_line[i]])
            # else:
            #     ans_list.append(ans_dict['_unk_'])
        ans_list.append(ans_dict['_eos_'])
        ans_len = len(ans_list)
        if ans_len < max_que_len:
            for line in range(max_que_len - ans_len):
                ans_list.append(ans_dict['_pad_'])
        else:
            ans_list = ans_list[:max_que_len - 1]
            ans_list.append(ans_dict['_eos_'])
        return ans_list

    def id2title(self, id_list, eos=3):
        result = ''
        for i in id_list:
            if i == eos:
                break
            result += self.dict_reverse[str(i)]
        return result


# 词汇类
content_vocab = Vocab('content_dict.txt')
title_vocab = Vocab('title_dict.txt')


# 测试功能
# title_vocab.title2id("自然语言处理")
# content_vocab.content2id("新闻标题生成任务")
# title_vocab.id2title([112, 4344, 5, 3])

# 数据集
# 读取文件
def read_file(file_name):
    content_list = []
    title_list = []
    with open(file_name, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            content, title = line.split("&")
            content_list.append(content)
            title_list.append(title)
    return content_list, title_list


# 数据基本格式
class FormatDatas(Dataset):
    def __init__(self, content_list, title_list, content_vocab=None, title_vocab=None):
        self.contents = content_list
        self.titles = title_list
        self.content_vocab = content_vocab
        self.title_vocab = title_vocab

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        title = self.titles[index]
        content = self.contents[index]
        if self.content_vocab:
            content = self.content_vocab.content2id(content)
        if self.title_vocab:
            title = self.title_vocab.title2id(title)
        return content, title


train_data = FormatDatas(*read_file(train_path), content_vocab, title_vocab)
val_data = FormatDatas(*read_file(val_path), content_vocab, title_vocab)


# seq2seq模型(transformer, 编码器、解码器、 自注意力机制)
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)

        self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward,
                                                   dropout, device) for _ in range(n_layers)])
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # scr = [batch size, src sent len]
        # src_mask = [batch size, src sent len]

        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)

        src = self.do((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src sent len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.In = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src sent len, hid dim]
        # src_mask = [batch size, src sent len]

        src = self.In(src + self.do(self.sa(src, src, src, src_mask)))

        src = self.In(src + self.do(self.pf(src)))

        return src


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = key = value = [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q = K = V = [batch size, sent len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # Q = K = V = [batch size, n heads, sent len, hid dim // n heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len, sent len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(torch.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len, sent len]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len, n heads, hid dim // n heads]

        x = x.view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len, hid dim]

        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = self.do(torch.relu(self.fc_1(x)))

        # x = [batch size, sent len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, sent len, hid dim]

        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)

        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward,
                                                   dropout, device) for _ in range(n_layers)])

        self.fc = nn.Linear(hid_dim, output_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, src, trg_mask, src_mask):
        # trg = [batch size, trg sent len]
        # src = [batch size, src sent len]
        # trg_mask = [batch size, trg sent len]
        # src_mask = [batch size, src sent len]

        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)

        trg = self.do((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch size, trg sent len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        return self.fc(trg)


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.In = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        # trg = [batch size, trg sent len, hid dim]
        # src = [batch size, src sent len, hid dim]
        # trg_mask = [batch size, trg sent len]
        # src_mask = [batch size, src sent len]

        trg = self.In(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.In(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.In(trg + self.do(self.pf(trg)))

        return trg


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx, pad_idx, device, max_len=70):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx = sos_idx
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.device = device

    def make_masks(self, src, trg):
        # src = [batch size, src sent len]
        # trg = [batch size, trg sent len]

        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

        # src = [batch size, 1, 1, src sent len]
        # trg_pad_mask = [batch size, 1, trg sent len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg sent len, trg sent len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg sent len, trg sent len]

        return src_mask, trg_mask

    def forward(self, src, trg):
        # src = [batch size, src sent len]
        # trg = [batch size, trg sent len]

        src_mask, trg_mask = self.make_masks(src, trg)

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src sent len, hid dim]

        out = self.decoder(trg, enc_src, trg_mask, src_mask)

        # out = [batch size, trg sent len, output dim]

        return out

    def translate_sentence(self, src):
        # src = [batch size, src sent len]
        # print(src)
        batch_size, src_len = src.shape
        trg = src.new_full((batch_size, 1), self.sos_idx)

        # trg = [batch size, 1]

        src_mask, trg_mask = self.make_masks(src, trg)

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src sent len, hid dim]

        translation_step = 0
        while translation_step < self.max_len:
            out = self.decoder(trg, enc_src, trg_mask, src_mask)

            # out - [batch size, trg sent len, output dim]

            out = torch.argmax(out[:, -1], dim=1)  # batch_size
            out = out.unsqueeze(1)  # batch_size, 1
            trg = torch.cat((trg, out), dim=1)

            # trg - [batch size, trg sent len]

            src_mask, trg_mask = self.make_masks(src, trg)
            translation_step += 1
        return trg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

output_max_len = 70
pad_idx = 0
sos_idx = 2

input_dim, output_dim = content_vocab.get_dict_dim(), title_vocab.get_dict_dim()
hid_dim = 128
n_layers = 3
n_heads = 8
pf_dim = 2048
dropout = 0.1

enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward,
              dropout, device)
dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward,
              dropout, device)
model = Seq2Seq(enc, dec, sos_idx, pad_idx, device, output_max_len).to(device)


# 训练阶段
# 参数统计
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")


# 损失函数、 优化器
# 定义optimizers
class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


optimizer = NoamOpt(hid_dim, 1, 2000, torch.optim.Adam(model.parameters(), lr=0.0000001, betas=(0.9, 0.98), eps=1e-9))
# 定义损失函数
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


# 分批生成器
def batch_iter(train_dataset, batch_size=32):
    data_len = len(train_dataset)
    num_batch = int((data_len - 1) / batch_size) + 1
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        src_batch = []
        trg_batch = []
        for j in range(start_id, end_id):
            # print(train_dataset[j][0])
            # print(train_dataset[j][1])
            src_batch.append(train_dataset[j][0])
            trg_batch.append(train_dataset[j][1])
        yield src_batch, trg_batch


# 训练步骤
def train(model, train_dataset, optimizer, criterion, clip, log_interval=100):
    model.train()
    epoch_loss = 0
    count = 0

    for i, batch_data in enumerate(batch_iter(train_dataset)):
        batchx, batchy = batch_data
        src = torch.tensor(batchx).to(device)
        trg = torch.tensor(batchy).to(device)
        optimizer.zero_grad()

        output = model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        count += 1

        epoch_loss += loss.item()
        if i % log_interval == 0:
            print(f"Train Step: {i} Loss: {loss.item():.3f}")

    return epoch_loss / count


# 验证步骤
def evaluate(model, valid_dataset, criterion):
    model.eval()
    epoch_loss = 0
    count = 0

    with torch.no_grad():
        for i, batch_data in enumerate(batch_iter(valid_dataset)):
            batchx, batchy = batch_data
            src = torch.tensor(batchx).to(device)
            trg = torch.tensor(batchy).to(device)

            output = model(src, trg[:, :-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            count += 1
    return epoch_loss / count


# import os
#
# if not os.path.exists('/home/zxy/model.pth'):
#     model.load_state_dict(torch.load('/home/zxy/model.pth'))


# 训练开始
clip = 1
best_valid_loss = float('inf')
losslist = []
epoches = 15

for epoch in range(epoches):
    print("Epochs: {}/{}".format(epoch + 1, epoches))
    train_loss = train(model, train_data, optimizer, criterion, clip)
    valid_loss = evaluate(model, val_data, criterion)
    print("Valid Loss: {:.3f}".format(valid_loss))
    losslist.append([train_loss, valid_loss])

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pth')


def predict(model, txt_seq):
    model.eval()
    # print(txt_seq)
    x_data = content_vocab.content2id(txt_seq)
    x = torch.tensor([x_data]).to(device)

    translation = model.translate_sentence(x)
    # print(translation)

    translation = translation[0].cpu().detach().numpy()
    res = title_vocab.id2title(translation[1:])
    return res


# 可视化loss变化过程
def draw_loss(losslist):
    x = []
    y = []
    for index, i in enumerate(losslist):
        x.append(index)
        y.append(i[0])
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='red', linewidth=1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.show()
    fig.savefig('loss.png', dpi=600, format='png')


draw_loss(losslist)


# # ROUGE评估
# def rouge_1(model, reference):  # term_reference是参考摘要，term_model是模型摘要， one-gram 一元模型
#     # jieba.load_userdict('/home/zxy/code/SportsSum/my_rewriter/data/linesup_group.txt')
#     terms_reference = jieba.cut(reference)  # 默认精准模式
#     terms_model = jieba.cut(model)
#     grams_reference = list(terms_reference)
#     grams_model = list(terms_model)
#     temp = 0
#     ngram_all = len(grams_reference)
#     for x in grams_reference:
#         if x in grams_model:
#             temp = temp + 1
#     rouge_1 = temp / ngram_all
#     return rouge_1
#
#
# def rouge_2(model, reference):  # term_reference是参考摘要，term_model是模型摘要， Bi-gram 二元模型
#     # jieba.load_userdict('/home/zxy/code/SportsSum/my_rewriter/data/linesup_group.txt')
#     terms_reference = jieba.cut(reference)  # 默认精准模式
#     terms_model = jieba.cut(model)
#     grams_reference = list(terms_reference)
#     grams_model = list(terms_model)
#     gram_2_model = []
#     gram_2_reference = []
#     temp = 0
#     ngram_all = len(grams_reference) - 1
#     for x in range(len(grams_model) - 1):
#         gram_2_model.append(grams_model[x] + grams_model[x + 1])
#     for x in range(len(grams_reference) - 1):
#         gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])
#     for x in gram_2_model:
#         if x in gram_2_reference:
#             temp = temp + 1
#     rouge_2 = temp / ngram_all
#     return rouge_2
#
#
# def rouge_score(model, reference):
#     rouge_1_score = rouge_1(model, reference)
#     rouge_2_score = rouge_2(model, reference)
#     print("Rouge-1: ", rouge_1_score, "||", "Rouge-2: ", rouge_2_score)
#     return rouge_1_score, rouge_2_score


# 绘制直方图
def draw_hist(data_list, name):
    fig = plt.figure(figsize=(8, 6), dpi=600)
    data_array = np.array(data_list)
    plt.hist(data_array, bins=10)
    fig.savefig(name, dpi=600, format='png')
    # plt.show()


# 选几条测评
def evaluation(test_path):
    title_list = []
    content_list = []
    score = []
    with open(test_path, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            content, title = line.split('&')
            title_list.append(title)
            content_list.append(content)
        for i in range(0, len(content_list)):
            live_pair = []
            news_pair = []
            predict_sentence = predict(model, content_list[i])
            live_pair.append(predict_sentence)
            news_pair.append(content_list[i])
            P, R, F1 = scorer.score(live_pair, news_pair)
            sc = float(F1)
            # rouge_1_score, rouge_2_score = rouge_score(predict_sentence, title_list[i])
            score.append(sc)
            print('原文：', content_list[i])
            print('参考句子：', title_list[i])
            print('预测句子：', predict_sentence, '\n')
            print('======================================')
    draw_hist(score, 'score_rouge.png')


evaluation(test_path)
