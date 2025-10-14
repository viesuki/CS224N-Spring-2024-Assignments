# %%
import numpy as np
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as todata
import time
import random
import math

# %%
# 1. 处理文本
class TextPreprocessor:
    def __init__(self, min_freq=4, sub_sampling=True):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.word_freq = []
        self.wf_dict = {}
        self.subsample_threshold = 2e-3
        self.subsampling = sub_sampling
        
    def split_sentences(self, text: str):
        """
        按句号/问号/感叹号/分号/冒号/换行来切分句子。
        注意：必须在清洗前做，否则标点被去掉就无法分句。
        """
        # 统一换行为空格，避免影响切分
        # text = text.replace('\n', ' ').replace('\r', ' ')
        # 用正则在句末标点后切分
        sentences = re.split(r'(?<=[\.\!\?\;\:\n])\s+', text)
        # 去掉空句子和过短句子
        sentences = [s.strip() for s in sentences if s and len(s.split())]
        return sentences
    
    def preprocess_text(self, text, subsampling=False):
        """清洗文本"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # 移除非字母字符
        words = text.split()

        def keep_word(word):
            # 下采样：随机丢弃高频词
            return (random.uniform(0, 1) <
                    math.sqrt(self.subsample_threshold / self.wf_dict[word]))
        
        if subsampling:
            words = [word for word in words if keep_word(word)]
        return words
    
    def sentences_to_indices(self, text):
        """
        先分句，再把每个句子清洗并映射成索引序列。
        返回: List[List[int]]，每个子列表是一句的索引序列
        """
        sentences = self.split_sentences(text)
        indices_per_sent = []
        for s in sentences:
            words = self.preprocess_text(s, subsampling=self.subsampling)   # 这里会去标点、转小写
            idxs = [self.word2idx.get(w, 0) for w in words]
            idxs = [i for i in idxs if i != 0]
            if idxs:                               # 保留非空句
                indices_per_sent.append(idxs)
        return indices_per_sent
    
    def build_vocab(self, corpus):
        """构建词汇表"""
        corpus = self.split_sentences(corpus)
        all_words = []
        for text in corpus:
            words = self.preprocess_text(text, subsampling=False)  # 构建词典不采样
            all_words.extend(words)
        
        # 统计词频，过滤低频词
        word_counts = Counter(all_words)
        filtered_words = [[word, count] for word, count in word_counts.items() 
                         if count >= self.min_freq]
        
        # 按词频从高到低排序
        self.word_freq = sorted(filtered_words, key=lambda x: x[1], reverse=True)

        total_words = sum([count for _, count in word_counts.items()])
        self.wf_dict = {word: count / total_words for word, count in word_counts.items()}
        
        # 创建词汇表映射
        self.word2idx = {'<UNK>': 0}
        self.idx2word = {0: '<UNK>'}
        
        for idx, [word, _] in enumerate(self.word_freq, 1):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        return self.word2idx, self.idx2word

# %%
# 2. 生成数据集
class DataGenerator:
    def __init__(self, word_freq, window_size=2, num_neg_samples=5):
        self.word_freq = word_freq
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples

        # 根据词频生成负采样分布（Word2Vec的采样策略）
        _, word_freq = zip(*word_freq)
        word_freq = np.array(word_freq)
        self.neg_distribution = torch.tensor(word_freq ** 0.75, dtype=torch.float32)
        self.neg_distribution /= self.neg_distribution.sum()
    
    def generate_skipgram_data(self, sent_indices):
        """生成Skip-gram训练数据"""
        centers, contexts = [], []
        for sent in sent_indices:
            if len(sent) <= 2: continue
            centers += [[w] for w in sent]
            for i in range(len(sent)):
                # 确定上下文窗口
                windows = random.randint(1, self.window_size)                
                indices = list(range(max(0, i-windows), min(len(sent), i+windows+1)))
                indices.remove(i)
                
                contexts.append([sent[idx] for idx in indices])
        return centers, contexts
    
    def generate_cbow_data(self):
        """生成CBOW训练数据"""
        data = []
        for indices in self.sent_indices:
            for i in range(self.window_size, len(indices) - self.window_size):
                center_word = indices[i]
                if center_word == 0:
                    continue

                context_words = []
                ok = True            
                # 收集上下文词
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j == i: 
                        continue
                    if indices[j] == 0:
                        ok = False; break
                    context_words.append(indices[j])
                if ok:
                    data.append((context_words, center_word))
        return data

    def get_negative_samples(self, exclude_indices):
        """生成负样本"""
        negative_samples = []
        def generator():
                return (torch.multinomial(
                    self.neg_distribution, 
                    len(indices) * self.num_neg_samples, 
                    replacement=True
                    )+1).tolist()
        
        for indices in exclude_indices:
            negs = generator()
            while [n for n in negs if n in indices]:
                negs = generator()
            negative_samples.append(negs)

        return negative_samples

    def batchify(self, data):
        """返回带负样本的batch"""
        max_len = max(len(c) + len(n) for _, c, n in data)
        centers, contexts_negatives, masks, labels = [], [], [], []
        for center, context, negative in data:
            cur_len = len(context) + len(negative)
            centers += [center]
            contexts_negatives += \
                [context + negative + [0] * (max_len - cur_len)]
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            labels += [[1] * len(context) + [0] * (max_len - len(context))]
        return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
            contexts_negatives), torch.tensor(masks), torch.tensor(labels))

    def load_data(self, sent_indices, model='skipgram', batch_size=32):
        centers, contexts = self.generate_skipgram_data(sent_indices)
        negatives = self.get_negative_samples(contexts)

        class CCNdata(todata.Dataset):
            def __init__(self, centers, contexts, negatives):
                self.centers = centers
                self.contexts = contexts
                self.negatives = negatives

            def __len__(self):
                return len(self.centers)

            def __getitem__(self, idx):
                return self.centers[idx], self.contexts[idx], self.negatives[idx]
            
        dataset = CCNdata(centers, contexts, negatives)
        # batchify = batchify(dataset)
        data_iter = todata.DataLoader(dataset, shuffle=True, 
                            collate_fn=self.batchify, batch_size=batch_size)
        
        return data_iter

        

    def generate_training_batches(self, sent_indices, model='skipgram', batch_size=32):
        """批量生成训练数据，减少内存占用"""
        if model == 'skipgram':
            data = self.generate_skipgram_data(sent_indices)
            data = torch.tensor(data, dtype=torch.long)
            print("skipgram data:", data.shape)
            perm = torch.randperm(len(data))
            data = data[perm]
        else:
            data = self.generate_cbow_data(sent_indices)
            print("CBOW data:", len(data))
            random.shuffle(data)
        for i in range(0, int(len(data)/batch_size)):
            yield data[i:i + batch_size]

# %%
# 3. 定义模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 中心词嵌入矩阵
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 上下文词嵌入矩阵  
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        bound = 0.5 / embedding_dim
        self.center_embeddings.weight.data.uniform_(-bound, bound)
        self.context_embeddings.weight.data.uniform_(-bound, bound)
    
    def forward(self, center_words, context_negatives):
        # 获取词向量
        v_center = self.center_embeddings(center_words)  # [batch_size, embedding_dim]
        u_context = self.context_embeddings(context_negatives)  # [batch_size, embedding_dim]
        print(v_center.shape, u_context.shape)

        return torch.bmm(v_center, u_context.transpose(1, 2))
    
    def get_word_vectors(self):
        """获取最终词向量（使用中心词矩阵）"""
        return self.center_embeddings.weight.data.cpu().numpy()
        

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size=2):
        super(CBOWModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.context_size = window_size * 2  # 左右各window_size个词
        
        # 上下文词嵌入矩阵
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 中心词嵌入矩阵
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        bound = 0.5 / embedding_dim
        self.context_embeddings.weight.data.uniform_(-bound, bound)
        self.center_embeddings.weight.data.uniform_(-bound, bound)
    
    def forward(self, context_words, center_words=None, negative_samples=None):
        """
        两种模式：
        1) 负采样：传入 center_words / negative_samples，返回 (positive_score, negative_score)
        2) 全词表：只传入 context_words，返回 [batch, vocab] 的打分（兼容旧用法）
        """
        # [batch, context_size, D]
        context_vectors = self.context_embeddings(context_words)
        # [batch, D] —— 平均上下文向量
        v_context = torch.mean(context_vectors, dim=1)

        if center_words is not None:
            # 负采样模式
            # 正样本打分
            u_pos = self.center_embeddings(center_words)       # [batch, D]
            positive_score = torch.sum(v_context * u_pos, dim=1)   # [batch]

            if negative_samples is not None:
                # [batch, num_neg, D]
                u_neg = self.center_embeddings(negative_samples)
                # [batch, num_neg, 1] = bmm([b,n,D], [b,D,1])
                negative_score = torch.bmm(u_neg, v_context.unsqueeze(2)).squeeze(2)  # [batch, num_neg]
                return positive_score, negative_score

            return positive_score

        # 兼容：全词表打分（原实现）
        u_center = self.center_embeddings.weight  # [vocab, D]
        scores = torch.mm(v_context, u_center.t())  # [batch, vocab]
        return scores
    
    def get_word_vectors(self):
        """获取最终词向量（使用上下文词矩阵）"""
        return self.context_embeddings.weight.data.cpu().numpy()   

# %%
# 4. 负采样
class NegativeSamplingTrainer:
    def __init__(self, word_freq, num_neg_samples=5):
        self.num_neg_samples = num_neg_samples
        
        # 根据词频生成负采样分布（Word2Vec的采样策略）
        _, word_freq = zip(*word_freq)
        word_freq = np.array(word_freq)
        self.neg_distribution = torch.tensor(word_freq ** 0.75, dtype=torch.float32)
        self.neg_distribution /= self.neg_distribution.sum()
    
    def get_negative_samples(self, exclude_indices=None, device='cpu'):
        """生成负样本"""
        negative_samples = []
        def generator():
                return (torch.multinomial(
                    self.neg_distribution, 
                    len(indices) * self.num_neg_samples, 
                    replacement=True
                    )+1).tolist()
        
        for indices in exclude_indices:
            negs = generator()
            while [n for n in negs if n in indices]:
                negs = generator()
            negative_samples.append(negs)

        return negative_samples
    
    def negative_sampling_loss(self, positive_score, negative_score):
        """负采样损失函数"""
        positive_loss = -F.logsigmoid(positive_score)
        negative_loss = -F.logsigmoid(-negative_score).sum(dim=1)
        
        return (positive_loss + negative_loss).mean()
# %%
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super(SigmoidBCELoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """带掩码的二元交叉熵损失函数"""
        loss = F.binary_cross_entropy_with_logits(inputs, 
                targets, weight=mask, reduction='mean')
        
        return loss



# %%
# 5. 训练模型
class Word2VecTrainer:
    def __init__(self, corpus, model_type='skipgram', embedding_dim=100, 
                 window_size=2, num_neg_samples=5, learning_rate=1e-3, device='cpu',
                 sub_sampling=False):
        self.corpus = corpus
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.device = device
        
        # 数据预处理
        self.preprocessor = TextPreprocessor(min_freq=2, sub_sampling=sub_sampling)
        self.word2idx, self.idx2word = self.preprocessor.build_vocab(corpus)
        self.vocab_size = self.preprocessor.vocab_size
        self.word_freq = self.preprocessor.word_freq
        
        # 数据生成
        self.data_generator = DataGenerator(self.word_freq,window_size=window_size)
        
        # 选择模型
        if model_type == 'skipgram':
            self.model = SkipGramModel(self.vocab_size, embedding_dim)
        else:  # CBOW
            self.model = CBOWModel(self.vocab_size, embedding_dim, window_size)
        
        # 训练器
        self.loss = SigmoidBCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, epochs=10, batch_size=32):
        """训练模型"""
        sent_indices_list = self.preprocessor.sentences_to_indices(self.corpus)
        print("数据处理完成，共有{}句子，平均长度为{:.2f}".
              format(len(sent_indices_list), np.mean([len(sent) for sent in sent_indices_list])))
        print("句子中单词 the 的总数为{}".format(sum(sent.count(self.preprocessor.word2idx['the']) for sent in sent_indices_list)))
        print("句子中单词 good 的总数为{}".format(sum(sent.count(self.preprocessor.word2idx['good']) for sent in sent_indices_list)))
        print("字典的大小为{}".format(self.vocab_size))

        self.model = self.model.to(self.device)
        self.model.train()
        
        data_iter = self.data_generator.load_data(sent_indices_list, batch_size)
        
        for epoch in range(epochs):
            total_loss, num_steps = 0, 0
            if self.model_type == 'skipgram':
                for batch in data_iter:
                    center, context_negative, mask, label = [
                        data.to(self.device) for data in batch]
                    
                    pred = self.model(center, context_negative)
                    
                    # 计算损失
                    loss = self.loss(pred.reshape(label.shape).float(), 
                                     label.float(), mask)
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_steps += 1

            else:  # CBOW
                for batch in data_iter:
                    if not batch:
                        continue

                    context_words = torch.tensor(
                        [d[0] for d in batch], dtype=torch.long, device=self.device
                    )  # [B, 2w]
                    center_words = torch.tensor(
                        [d[1] for d in batch], dtype=torch.long, device=self.device
                    )  # [B]

                    # 负采样（建议已实现索引 +1 对齐 & 排除正样本）
                    negative_samples = self.trainer.get_negative_samples(
                        batch_size=len(batch),
                        exclude_indices=center_words,
                        device=self.device
                    )

                    # 前向（返回正/负样本打分）
                    positive_score, negative_score = self.model(
                        context_words, center_words, negative_samples
                    )

                    # 负采样损失（建议使用 logsigmoid 版本）
                    loss = self.trainer.negative_sampling_loss(positive_score, negative_score)


                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_steps += 1
            
            avg_loss = total_loss / max(1, num_steps)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    def get_word_vectors(self):
        """获取训练好的词向量"""
        return self.model.get_word_vectors()
    
    def find_similar_words(self, word, top_k=5):
        """查找相似词"""
        if word not in self.word2idx:
            return f"单词 '{word}' 不在词汇表中"
        
        all_vecs = torch.tensor(self.model.get_word_vectors())
        word_idx = self.word2idx[word]
        word_vec = all_vecs[word_idx]
        
        # 计算与所有词的余弦相似度
        sims = torch.cosine_similarity(word_vec.unsqueeze(0), all_vecs, dim=1)
        sims[word_idx] = -1e9           # 排除自身
        sims[0] = -1e9                  # 排除 <UNK>

        top_indices = torch.topk(sims, top_k).indices.tolist()
        return [(self.idx2word[i], sims[i].item()) for i in top_indices]
    
    def word2vec(self, word):
        """将单词转换为向量"""
        if word not in self.word2idx:
            return None
        index =  self.word2idx[word]
        return self.model.get_word_vectors()[index]
    
    def vec2word(self, vector):
        """将向量转换为单词"""
        all_vecs = torch.tensor(self.model.get_word_vectors())  # [V, D]
        q = torch.tensor(vector).unsqueeze(0)                   # [1, D]
        sims = torch.cosine_similarity(q, all_vecs, dim=1)
        sims[0] = -1e9
        top_index = sims.argmax().item()
        return self.idx2word[top_index]

# %%

def read_text(text):  #@save
    """将数据集加载到文本行的列表中"""
    with open(text, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read()
    return lines

# %%
# text = 'shakespeare.txt'
text = 'ptb.train.txt'
# text = 'timemachine.txt'
corpus = read_text(text)

# %%
stime = time.time()
# 训练Skip-gram模型
print("训练Skip-gram模型...")
skipgram_trainer = Word2VecTrainer(
    corpus=corpus,
    model_type='skipgram',
    embedding_dim=50,
    window_size=3,
    learning_rate=2e-4,
    device='cuda',
    sub_sampling=True,
)
skipgram_trainer.train(epochs=50, batch_size=64)
etime = time.time()
print(f'训练完成，总耗时: {etime - stime:.2f} 秒； \n 平均耗时: {(etime - stime) / 50:.2f} 秒')

# %%
# stime = time.time()
# # 训练CBOW模型
# print("训练CBOW模型...")
# cbow_trainer = Word2VecTrainer(
#     corpus=corpus,
#     model_type='cbow',
#     embedding_dim=50,
#     window_size=2,
#     learning_rate=2e-3,
#     device='cuda',
#     sub_sampling=True,
# )
# cbow_trainer.train(epochs=5, batch_size=32)
# etime = time.time()
# print(f'训练完成，总耗时: {etime - stime:.2f} 秒； \n 平均耗时: {(etime - stime) / 50:.2f} 秒')

# %%
len(skipgram_trainer.word2idx), len(skipgram_trainer.word_freq)

# %%
skipgram_trainer.vec2word(skipgram_trainer.get_word_vectors()[1])

# %%
skipgram_trainer.find_similar_words('poor'), skipgram_trainer.find_similar_words('chip')
#, cbow_trainer.find_similar_words('great')
# # %%
# cbow_trainer.vec2word(cbow_trainer.get_word_vectors()[1])

# %%
word = 'blood'
skipgram_trainer.word2vec(word) == skipgram_trainer.get_word_vectors()[skipgram_trainer.word2idx[word]]

# %%


# %%
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def reduce_to_k_dim(M, k=2):
    n_iters = 10 
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    M_reduced = TruncatedSVD(n_components=k, n_iter=n_iters).fit_transform(M)
    return M_reduced

def plot_embeddings(M_reduced, word2ind, words):
    # simulating a pandas df['type'] column
    num_plot = min(30, len(words))
    types = [word for word in words][:num_plot]
    x_coords = [M_reduced[word2ind[word], 0] for word in types]
    y_coords = [M_reduced[word2ind[word], 1] for word in types]

    for i,type in enumerate(types):
        x = x_coords[i]
        y = y_coords[i]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, type, fontsize=9)
    plt.show()
# %%
word = ['humor', 'good', 'great', 'fun', 'strong', 'thanks', 'he', 'she', 
        'you', 'girl', 'boy', 'them', 'laugh', 'film', 'movie', 
        'emotional', 'fine', 'wonderful', 'poor', 'gain', 'barren', 'blood',
        'thee', 'sake', 'blush', 'speak', 'greater', 'woman', 'man']
word = [w for w in word if w in skipgram_trainer.word2idx]


M_reduced = reduce_to_k_dim(skipgram_trainer.get_word_vectors())
plot_embeddings(M_reduced, skipgram_trainer.word2idx, word)
# %%
# M_reduced = reduce_to_k_dim(cbow_trainer.get_word_vectors())
# plot_embeddings(M_reduced, cbow_trainer.word2idx, word)
# %%
print("词向量类比测试：'he'-'she' + 'woman' = ?", 
        skipgram_trainer.vec2word(skipgram_trainer.word2vec('he')-skipgram_trainer.word2vec('she')
       + skipgram_trainer.word2vec('woman')))

# %%


# %%
# print("词向量类比测试：'he'-'she' + 'woman' = ?", 
#         cbow_trainer.vec2word(cbow_trainer.word2vec('he')-cbow_trainer.word2vec('she')
#        + cbow_trainer.word2vec('woman')))

# %%