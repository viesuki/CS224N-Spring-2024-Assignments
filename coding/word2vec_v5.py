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
    def __init__(self, min_freq=2, sub_sampling=True):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.word_freq = []
        self.wf_dict = {}
        self.subsample_threshold = 1e-3
        self.subsampling = sub_sampling
        
    def split_sentences(self, text: str):
        """
        按句号/问号/感叹号/分号/冒号/换行来切分句子。
        注意：必须在清洗前做，否则标点被去掉就无法分句。
        """
        # 用正则在句末标点后切分
        sentences = re.split(r'(?<=[\.\!\?\;\:\n\r])\s+', text)
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
            f = self.wf_dict.get(word, 1e-12)   # 稀有词近似必保留
            return (random.uniform(0, 1) < 
                    math.sqrt(self.subsample_threshold / f))
        
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
    
    def generate_cbow_data(self, sent_indices):
        """生成CBOW训练数据"""
        centers, contexts = [], []
        for sent in sent_indices:
            windows = random.randint(1, self.window_size)
            if len(sent) <= windows * 2: continue
            for i in range(windows, len(sent) - windows):
                centers += [[sent[i]]]
                contexts += [sent[i - windows:i] + sent[i + 1:i + windows + 1]]

        return centers, contexts

    def get_negative_samples(self, exclude_indices):
        """生成负样本"""
        negative_samples = []
        for indices in exclude_indices:
            # 批量生成，避免重复采样
            negs = (torch.multinomial(
                self.neg_distribution, 
                len(indices) * self.num_neg_samples * 2,  # 多采样一些
                replacement=True
            ) + 1).tolist()
            
            # 过滤掉出现在上下文中的词
            valid_negs = [n for n in negs if n not in indices]
            # 取所需数量
            negative_samples.append(valid_negs[:len(indices) * self.num_neg_samples])
        
        return negative_samples

    def batchify(self, data):
        """返回带负样本的batch"""
        max_len = max(len(c) + len(n) for _, c, n in data)
        max_cen = max(len(c) for c, _, _ in data)
        centers,mask_c, contexts_negatives, mask_n, labels = [], [], [], [], []
        for center, context, negative in data:
            cur_len = len(context) + len(negative)
            centers += [center+[0]*(max_cen-len(center))]
            mask_c += [len(center)]
            contexts_negatives += \
                [context + negative + [0] * (max_len - cur_len)]
            mask_n += [[1] * cur_len + [0] * (max_len - cur_len)]
            labels += [[1] * len(context) + [0] * (max_len - len(context))]
        return (torch.tensor(centers), torch.tensor(mask_c) ,torch.tensor(
            contexts_negatives), torch.tensor(mask_n), torch.tensor(labels))
    

    def load_data(self, sent_indices, model='skipgram', batch_size=32):
        if model == 'skipgram':
            centers, contexts = self.generate_skipgram_data(sent_indices)
            negatives = self.get_negative_samples(contexts)
        else:
            contexts, centers = self.generate_cbow_data(sent_indices)  # 给他俩调换位置
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
        print("训练数据条目：{}".format(len(dataset)))
        # batchify = batchify(dataset)
        data_iter = todata.DataLoader(dataset, shuffle=True, 
                            collate_fn=self.batchify, batch_size=batch_size,
                            pin_memory=torch.cuda.is_available())
        
        return data_iter

# %%
# 3. 定义模型
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 中心词嵌入矩阵
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 上下文词嵌入矩阵  
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 初始化权重
        self._init_weights()
    
    def forward(self, center_words, context_negatives, mask=None):
        # 获取词向量
        v_center = self.center_embeddings(center_words)  # [batch_size, 1, embedding_dim]
        u_context = self.context_embeddings(context_negatives)  # [batch_size, max_len, embedding_dim]
                    
        return torch.bmm(v_center, u_context.transpose(1, 2))
    
    def get_word_vectors(self):
        """获取最终词向量（使用中心词矩阵）"""
        return self.center_embeddings.weight.detach().cpu().numpy()
    
    def _init_weights(self):
        """统一的权重初始化"""
        bound = 1.0 / math.sqrt(self.embedding_dim)
        for embedding in [self.center_embeddings, self.context_embeddings]:
            nn.init.uniform_(embedding.weight, -bound, bound)
            # 固定UNK向量的梯度
            embedding.weight.data[0] = 0
        
# %%
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 上下文词嵌入矩阵
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 中心词嵌入矩阵
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 初始化权重
        self._init_weights()
    
    def forward(self, center_words, context_negatives, mask_c=None):
        # [batch, context_size, D]
        v_context = self.context_embeddings(center_words)
        # [batch, D] —— 平均上下文向量
        v_context = torch.sum(v_context, dim=1) / mask_c.unsqueeze(1)
        u_center = self.center_embeddings(context_negatives)

        return torch.bmm(v_context.unsqueeze(1), u_center.transpose(1, 2))
    
    def get_word_vectors(self):
        """获取最终词向量（使用上下文词矩阵）"""
        return self.context_embeddings.weight.detach().cpu().numpy()
    
    def _init_weights(self):
        """统一的权重初始化"""
        bound = 1.0 / math.sqrt(self.embedding_dim)
        for embedding in [self.center_embeddings, self.context_embeddings]:
            nn.init.uniform_(embedding.weight, -bound, bound)
            # 固定UNK向量的梯度
            embedding.weight.data[0] = 0

# %%
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super(SigmoidBCELoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """带掩码的二元交叉熵损失函数"""
        if mask is not None:
            # 确保mask与inputs形状一致
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            # 应用mask
            inputs = inputs * mask
            targets = targets * mask
        
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='sum'
        )
        
        # 如果使用mask，按有效元素数归一化
        if mask is not None:
            loss = loss / mask.sum().clamp(min=1)
        else:
            loss = loss / inputs.numel()
            
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
            self.model = CBOWModel(self.vocab_size, embedding_dim)
        
        # 训练器
        self.loss = SigmoidBCELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
    def train(self, epochs=10, batch_size=32, eval_words=None):
        """训练模型"""
        # 在训练类中添加评估词
        if eval_words is None:
            eval_words = ['good', 'great', 'man', 'woman']

        sent_indices_list = self.preprocessor.sentences_to_indices(self.corpus)
        print("数据处理完成，共有 {} 句子，平均长度为 {:.2f}".
              format(len(sent_indices_list), np.mean([len(sent) for sent in sent_indices_list])))
        print("句子中单词 the 的采样数 {}".format(sum(sent.count(self.preprocessor.word2idx['the']) for sent in sent_indices_list)))
        print("句子中单词 good 的采样数 {}".format(sum(sent.count(self.preprocessor.word2idx['good']) for sent in sent_indices_list)))

        self.model = self.model.to(self.device)
        self.model.train()
        
        data_iter = self.data_generator.load_data(sent_indices_list, 
                                                  self.model_type, batch_size)
        
        for epoch in range(epochs):
            total_loss, num_steps = 0, 0
            for batch in data_iter:
                center, mask_c , context_negative, mask_n, label = [
                    data.to(self.device) for data in batch]
                
                pred = self.model(center, context_negative, mask_c)
                
                # 计算损失
                loss = self.loss(pred.reshape(label.shape).float(), 
                                    label.float(), mask_n)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_steps += 1
            
            avg_loss = total_loss / max(1, num_steps)
            # 每5个epoch结束后评估相似词            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
            if (epoch + 1) % 5 == 0:
                for word in eval_words:
                    if word in self.word2idx:
                        similar = self.find_similar_words(word, top_k=3)
                        print(f"  '{word}' -> {similar}")
    
    def get_word_vectors(self):
        """获取训练好的词向量"""
        return self.model.get_word_vectors()
    
    def find_similar_words(self, word, top_k=5):
        """查找相似词"""
        if word not in self.word2idx:
            return f"单词 '{word}' 不在词汇表中"
        
        all_vecs = torch.from_numpy(self.model.get_word_vectors())
        word_idx = self.word2idx[word]
        word_vec = all_vecs[word_idx]
        
        # L2 归一化相似度更稳
        all_vecs = F.normalize(all_vecs, dim=1)
        w = F.normalize(word_vec.unsqueeze(0), dim=1).squeeze(0)
        sims = torch.mv(all_vecs, w)  # [V]
        sims[word_idx] = -1e9
        sims[0] = -1e9

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
        all_vecs = torch.from_numpy(self.model.get_word_vectors())  # [V, D]
        w = torch.from_numpy(vector).unsqueeze(0)
        all_vecs = F.normalize(all_vecs, dim=1)
        w = F.normalize(w, dim=1).squeeze(0)
        sims = torch.mv(all_vecs, w)
        sims[0] = -1e9
        top_index = sims.argmax().item()
        return self.idx2word[top_index]
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'config': {
                'embedding_dim': self.embedding_dim,
                'vocab_size': self.vocab_size,
                'model_type': self.model_type
            }
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath, device='cpu'):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        # 重新创建训练器实例
        trainer = cls.__new__(cls)
        trainer.word2idx = checkpoint['word2idx']
        trainer.idx2word = checkpoint['idx2word']
        trainer.vocab_size = checkpoint['config']['vocab_size']
        trainer.embedding_dim = checkpoint['config']['embedding_dim']
        trainer.model_type = checkpoint['config']['model_type']
        
        # 重新创建模型
        if trainer.model_type == 'skipgram':
            trainer.model = SkipGramModel(trainer.vocab_size, trainer.embedding_dim)
        else:
            trainer.model = CBOWModel(trainer.vocab_size, trainer.embedding_dim)
        
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.to(device)
        return trainer

# %%

def read_text(text):  #@save
    """将数据集加载到文本行的列表中"""
    with open(text, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read()
    return lines

# %%
# text = 'shakespeare.txt'
# text1 = 'ptb.train.txt'
text = ['shakespeare.txt', 'ptb.train.txt']
# text = 'timemachine.txt'
corpus = ''
for t in text:
    corpus += read_text(t)

# %%
stime = time.time()
# 训练Skip-gram模型
print("训练 Skip-gram 模型...")
skipgram_trainer = Word2VecTrainer(
    corpus=corpus,
    model_type='skipgram',
    embedding_dim=50,
    window_size=2,
    learning_rate=2e-4,
    device='cuda',
    sub_sampling=True,
)
skipgram_trainer.train(epochs=50, batch_size=64)
etime = time.time()
print(f'训练完成，总耗时: {etime - stime:.2f} 秒； \n 平均耗时: {(etime - stime) / 50:.2f} 秒')
skipgram_trainer.save_model('skipgram.pt')

# %%
stime = time.time()
# 训练CBOW模型
print("训练 CBOW 模型...")
cbow_trainer = Word2VecTrainer(
    corpus=corpus,
    model_type='cbow',
    embedding_dim=50,
    window_size=2,
    learning_rate=4e-4,
    device='cuda',
    sub_sampling=True,
)
cbow_trainer.train(epochs=50, batch_size=64)
etime = time.time()
print(f'训练完成，总耗时: {etime - stime:.2f} 秒； \n 平均耗时: {(etime - stime) / 50:.2f} 秒')
cbow_trainer.save_model('cbow.pt')

# %%
len(skipgram_trainer.word2idx), len(skipgram_trainer.word_freq)

# %%
skipgram_trainer.vec2word(skipgram_trainer.get_word_vectors()[1])

# %%
skipgram_trainer.find_similar_words('good'), cbow_trainer.find_similar_words('great')
# %%
cbow_trainer.vec2word(cbow_trainer.get_word_vectors()[1])

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
word = [w for w in word if w in cbow_trainer.word2idx]
M_reduced = reduce_to_k_dim(cbow_trainer.get_word_vectors())
plot_embeddings(M_reduced, cbow_trainer.word2idx, word)
# %%
print("词向量类比测试：'boy'-'girl' + 'woman' = ?", 
        skipgram_trainer.vec2word(skipgram_trainer.word2vec('boy')-skipgram_trainer.word2vec('girl')
       + skipgram_trainer.word2vec('woman')))

# %%


# %%
print("词向量类比测试：'he'-'she' + 'woman' = ?", 
        cbow_trainer.vec2word(cbow_trainer.word2vec('he')-cbow_trainer.word2vec('she')
       + cbow_trainer.word2vec('woman')))

# %%
skipgram_trainer.find_similar_words('chip')

# %%
cbow_trainer.find_similar_words('chip')
# %%
len(skipgram_trainer.get_word_vectors()), len(skipgram_trainer.idx2word), skipgram_trainer.word2vec(skipgram_trainer.vec2word(skipgram_trainer.word2vec('chip'))) == skipgram_trainer.get_word_vectors()[skipgram_trainer.word2idx['chip']]

# %%
