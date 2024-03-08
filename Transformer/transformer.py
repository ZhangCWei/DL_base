import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""===参数设置==="""
d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8


"""===数据准备==="""
# 训练数据. 其中, P: 占位符号  E: 结束符号  S: 开始符号
# 三列分别是 Encoder_input, Decoder_input, Decoder_output
sentences = [['我 是 学 生 P' , 'S I am a student'   , 'I am a student E'],
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],
             ['我 是 男 生 P' , 'S I am a boy'       , 'I am a boy E']]

# 词源字典
src_vocab = {'P':0, '我':1, '是':2, '学':3, '生':4, '喜':5, '欢':6,'习':7,'男':8}   # 词源字典  字：索引
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)

# 目标字典
tgt_vocab = {'P':0, 'S':1, 'E':2, 'I':3, 'am':4, 'a':5, 'student':6, 'like':7, 'learning':8, 'boy':9}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
tgt_vocab_size = len(tgt_vocab)

# Encoder输入的最大长度
src_len = len(sentences[0][0].split(" "))

# Decoder输入输出最大长度
tgt_len = len(sentences[0][1].split(" "))


# 把 sentences 转换成字典索引
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)


# 自定义数据集类
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)


"""===定义位置信息==="""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 初始化一个dropout层，用于减少过拟合
        self.dropout = nn.Dropout(p=dropout)
        # 创建位置编码表
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
             if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        # 对偶数位置使用正弦函数进行编码
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])       # 字嵌入维度为偶数时
        # 对奇数位置使用余弦函数进行编码
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])       # 字嵌入维度为奇数时
        # 将位置编码表转换为张量，并传输到GPU上（如果可用）
        self.pos_table = torch.FloatTensor(pos_table).to(device)
    def forward(self, enc_inputs):                              # enc_inputs: [batch_size, seq_len, d_model]
        # 将位置编码添加到输入编码上
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        # 应用dropout并返回结果
        return self.dropout(enc_inputs.to(device))


"""===padding mask==="""
def get_attn_pad_mask(seq_q, seq_k):
    # 获取输入序列的批次大小和长度
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 创建一个掩码变量，该变量将输入序列中的所有填充元素（值为0的元素）标记为1，其他为0
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k]
    # 将掩码张量扩展到所需的尺寸。现在它的尺寸是[batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


"""===sequence mask==="""
def get_attn_subsequence_mask(seq):
    # 确定需要的掩码张量的形状
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 创建一个上三角矩阵，其中对角线以上的元素为1，其余为0
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # [batch_size, tgt_len, tgt_len]
    # 将NumPy数组转换为PyTorch张量，并确保其数据类型为byte（布尔类型）
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


"""===计算注意力信息、残差和归一化==="""
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)    # 计算注意力得分
        scores.masked_fill_(attn_mask, -1e9)                            # 应用注意力遮罩（用于填充或防止未来信息泄露）
        attn = nn.Softmax(dim=-1)(scores)                               # 应用softmax获取注意力权重 [batch_size, n_heads, len_q, len_k]
        context = torch.matmul(attn, V)                                 # 用注意力权重乘以值张量得到上下文张量
        return context, attn                                            # 返回上下文张量和注意力权重


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)         # 用于合并多头输出的最终线性层

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        # 为多头注意力投影并重塑输入Q, K, V
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)              # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # 从ScaledDotProductAttention获取上下文和注意力权重
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)          # context: [batch_size, n_heads, len_q, d_v]
        # 为最终线性层重塑上下文                                                     # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                # [batch_size, len_q, d_model]
        # 应用层归一化并添加残差连接
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


"""===前馈神经网络==="""
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            # 定义前馈全连接网络：两个线性层和一个ReLU激活层
            nn.Linear(d_model, d_ff, bias=False),   # 第一个线性层，从d_model维映射到d_ff维
            nn.ReLU(),                              # ReLU激活函数
            nn.Linear(d_ff, d_model, bias=False))   # 第二个线性层，从d_ff维映射回d_model维

    def forward(self, inputs):          # inputs: [batch_size, seq_len, d_model]
        residual = inputs               # 保留输入作为残差连接
        output = self.fc(inputs)        # 通过前馈网络传递输入
        return nn.LayerNorm(d_model).to(device)(output + residual)


"""===encoder==="""
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                                     # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # 然后将多头注意力的输出传递给位置前馈网络
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


"""===Encoder==="""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 源语言嵌入层，将源语言转换为词向量
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 位置编码层，增加位置信息
        self.pos_emb = PositionalEncoding(d_model)
        # 多层 EncoderLayer
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        # 将输入单词索引转换为嵌入向量
        enc_outputs = self.src_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        # 加入位置编码
        enc_outputs = self.pos_emb(enc_outputs)                                  # enc_outputs: [batch_size, src_len, d_model]
        # 生成自注意力层mask
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        # 存储每个编码器层的自注意力权重
        enc_self_attns = []
        for layer in self.layers:
            # 依次通过每个编码器层
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        # 返回最终输出和所有编码器层自注意力权重
        return enc_outputs, enc_self_attns


"""===decoder==="""
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # dec_inputs: [batch_size, tgt_len, d_model]
        # enc_outputs: [batch_size, src_len, d_model]
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


"""===Decoder==="""
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # dec_inputs: [batch_size, tgt_len]
        # enc_intpus: [batch_size, src_len]
        # enc_outputs: [batsh_size, src_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)                                            # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs).to(device)                                # [batch_size, tgt_len, d_model]
        # 创建解码器自注意力层的mask(包括padding和sequence)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)     # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device) # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device)   # [batch_size, tgt_len, tgt_len]
        # 创建解码器-编码器注意力层的mask
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)                                            # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # 通过每个解码器层
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # 返回解码器的输出和注意力权重
        return dec_outputs, dec_self_attns, dec_enc_attns


"""===Transformer==="""
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # 初始化编码器
        self.Encoder = Encoder().to(device)
        # 初始化解码器
        self.Decoder = Decoder().to(device)
        # 初始化输出投影层
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)
    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs: [batch_size, src_len]
        # dec_inputs: [batch_size, tgt_len]

        # 通过编码器处理输入
        # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)

        # 通过解码器处理编码器的输出和解码器的输入
        # dec_outpus    : [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(dec_inputs, enc_inputs, enc_outputs)

        # 将解码器的输出通过一个线性层转换为最终的输出词汇概率分布
        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)

        # 返回解码器输出的词汇概率分布以及编码器和解码器的注意力权重
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


model = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)                     # 定义交叉熵损失函数, 忽略索引为0的占位符
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)   # 随机梯度下降优化器


"""===训练==="""
for epoch in range(50):  # 训练50个周期
    for enc_inputs, dec_inputs, dec_outputs in loader:  # 从数据加载器中获取数据
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        # 清零梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新所有参数
        optimizer.step()


"""===测试==="""
def test(model, enc_input, start_symbol):
    # 使用模型的编码器部分处理输入序列并获得编码器输出
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    # 初始化解码器输入为全零张量，长度等于目标序列长度
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    # 设置起始符号，通常是序列的开始标记
    next_symbol = start_symbol

    # 逐步生成目标序列的每个词
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol  # 更新解码器输入的当前位置为最新生成的符号
        # 使用模型的解码器部分和当前的解码器输入生成下一个输出
        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)
        # 通过输出投影层将解码器输出转换为词汇分布
        projected = model.projection(dec_outputs)
        # 选择概率最高的词作为下一个符号
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]  # 获取当前位置的预测词
        next_symbol = next_word.item()  # 更新下一个符号

    # 返回构建的完整目标序列
    return dec_input


# 从数据加载器获取一个批次的输入
enc_inputs, _, _ = next(iter(loader))
# 使用测试函数和第一个编码器输入生成预测的目标序列
predict_dec_input = test(model, enc_inputs[0].view(1, -1).to(device), start_symbol=tgt_vocab["S"])
# 使用模型和预测的目标序列获得最终的预测输出
predict, _, _, _ = model(enc_inputs[0].view(1, -1).to(device), predict_dec_input)
# 选择预测结果中概率最高的词
predict = predict.data.max(1, keepdim=True)[1]

# 打印原始输入序列和模型生成的目标序列
print([src_idx2word[int(i)] for i in enc_inputs[0]], '->',
      [idx2word[n.item()] for n in predict.squeeze()])
