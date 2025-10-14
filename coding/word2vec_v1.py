# %%
import numpy as np
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import time

# %%
# 1. 处理文本
class TextPreprocessor:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.word_freq = []
        
    def preprocess_text(self, text):
        """清洗文本"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # 移除非字母字符
        words = text.split()
        return words
    
    def build_vocab(self, corpus):
        """构建词汇表"""
        all_words = []
        for text in corpus:
            words = self.preprocess_text(text)
            all_words.extend(words)
        
        # 统计词频，过滤低频词
        word_counts = Counter(all_words)
        # self.word_freq = sorted(word_counts.items(), key=lambda x: x[1] if x[1] 
        #                         >= self.min_freq else None, reverse=True)
        filtered_words = [[word, count] for word, count in word_counts.items() 
                         if count >= self.min_freq]
        self.word_freq = sorted(filtered_words, key=lambda x: x[1], reverse=True)
        
        # 创建词汇表映射
        self.word2idx = {'<UNK>': 0}
        self.idx2word = {0: '<UNK>'}
        
        for idx, word in enumerate(self.word_freq, 1):
            self.word2idx[word[0]] = idx
            self.idx2word[idx] = word[0]
        
        self.vocab_size = len(self.word2idx)
        return self.word2idx, self.idx2word
    
    def text_to_indices(self, text):
        """将文本转换为索引序列"""
        words = self.preprocess_text(text)
        indices = [self.word2idx.get(word, 0) for word in words]  # 0代表UNK
        return indices

# %%
# 2. 生成数据集
class DataGenerator:
    def __init__(self, window_size=2):
        self.window_size = window_size
    
    def generate_skipgram_data(self, indices):
        """生成Skip-gram训练数据"""
        data = []
        for i, center_word in enumerate(indices):
            # 确定上下文窗口
            start = max(0, i - self.window_size)
            end = min(len(indices), i + self.window_size + 1)
            
            for j in range(start, end):
                if j != i:  # 跳过中心词本身
                    context_word = indices[j]
                    data.append((center_word, context_word))
        return data
    
    def generate_cbow_data(self, indices):
        """生成CBOW训练数据"""
        data = []
        for i in range(self.window_size, len(indices) - self.window_size):
            center_word = indices[i]
            context_words = []
            
            # 收集上下文词
            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i:
                    context_words.append(indices[j])
            
            data.append((context_words, center_word))
        return data

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
        self.center_embeddings.weight.data.uniform_(-1, 1)
        self.context_embeddings.weight.data.uniform_(-1, 1)
    
    def forward(self, center_words, context_words, negative_samples=None):
        # 获取词向量
        v_center = self.center_embeddings(center_words)  # [batch_size, embedding_dim]
        u_context = self.context_embeddings(context_words)  # [batch_size, embedding_dim]
        
        # 计算正样本的得分
        positive_score = torch.sum(v_center * u_context, dim=1)  # [batch_size]
        positive_score = torch.clamp(positive_score, max=10, min=-10)
        
        # 负采样损失
        if negative_samples is not None:
            u_negative = self.context_embeddings(negative_samples)  # [batch_size, num_neg, embedding_dim]
            negative_score = torch.bmm(u_negative, v_center.unsqueeze(2))  # [batch_size, num_neg, 1]
            negative_score = torch.clamp(negative_score, max=10, min=-10)
            return positive_score, negative_score.squeeze(2)
        
        return positive_score
    
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
        self.context_embeddings.weight.data.uniform_(-1, 1)
        self.center_embeddings.weight.data.uniform_(-1, 1)
    
    def forward(self, context_words, negative_samples=None):
        # 获取所有上下文词的嵌入
        context_vectors = self.context_embeddings(context_words)  # [batch_size, context_size, embedding_dim]
        
        # 平均上下文向量（CBOW的核心）
        v_context = torch.mean(context_vectors, dim=1)  # [batch_size, embedding_dim]
        
        # 中心词向量（用于计算得分）
        u_center = self.center_embeddings.weight  # [vocab_size, embedding_dim]
        
        # 计算所有可能中心词的得分
        scores = torch.mm(v_context, u_center.t())  # [batch_size, vocab_size]
        
        return scores
    
    def get_word_vectors(self):
        """获取最终词向量（使用上下文词矩阵）"""
        return self.context_embeddings.weight.data.cpu().numpy()   

# %%
# 4. 负采样
class NegativeSamplingTrainer:
    def __init__(self, model, vocab_size, word_freq, num_neg_samples=5):
        self.model = model
        self.vocab_size = vocab_size
        self.num_neg_samples = num_neg_samples
        
        # 根据词频生成负采样分布（Word2Vec的采样策略）
        # word_freq = np.ones(vocab_size)  # 简化版，实际应该用真实词频
        _, word_freq = zip(*word_freq)
        word_freq = np.array(word_freq)
        self.neg_distribution = torch.tensor(word_freq ** 0.75, dtype=torch.float32)
        self.neg_distribution /= self.neg_distribution.sum()
    
    def get_negative_samples(self, batch_size, exclude_indices):
        """生成负样本"""
        negative_samples = torch.multinomial(
            self.neg_distribution, 
            batch_size * self.num_neg_samples, 
            replacement=True
        ).view(batch_size, self.num_neg_samples)
        return negative_samples
    
    def negative_sampling_loss(self, positive_score, negative_score):
        """负采样损失函数"""
        positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-10)
        negative_loss = -torch.log(1 - torch.sigmoid(negative_score) + 1e-10)
        
        return (positive_loss + negative_loss.sum(dim=1)).mean()

# %%
# 5. 训练模型
class Word2VecTrainer:
    def __init__(self, corpus, model_type='skipgram', embedding_dim=100, 
                 window_size=2, num_neg_samples=5, learning_rate=0.025, device='cpu'):
        self.corpus = corpus
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.device = device
        
        # 数据预处理
        self.preprocessor = TextPreprocessor(min_freq=2)
        self.word2idx, self.idx2word = self.preprocessor.build_vocab(corpus)
        self.vocab_size = self.preprocessor.vocab_size
        self.word_freq = self.preprocessor.word_freq
        
        # 数据生成
        self.data_generator = DataGenerator(window_size=window_size)
        
        # 选择模型
        if model_type == 'skipgram':
            self.model = SkipGramModel(self.vocab_size, embedding_dim)
        else:  # CBOW
            self.model = CBOWModel(self.vocab_size, embedding_dim, window_size)
        
        # 训练器
        self.trainer = NegativeSamplingTrainer(self.model, self.vocab_size, 
                                               self.word_freq, num_neg_samples)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
    
    def prepare_training_data(self):
        """准备训练数据"""
        all_training_data = []
        
        for text in self.corpus:
            indices = self.preprocessor.text_to_indices(text)
            if len(indices) < self.window_size * 2 + 1:
                continue
                
            if self.model_type == 'skipgram':
                data = self.data_generator.generate_skipgram_data(indices)
            else:  # CBOW
                data = self.data_generator.generate_cbow_data(indices)
            
            all_training_data.extend(data)
        
        return torch.tensor(np.array(all_training_data))
    
    def train(self, epochs=10, batch_size=32):
        """训练模型"""
        training_data = self.prepare_training_data()
        print(f"训练数据数量: {len(training_data)}")
        
        self.model = self.model.to(self.device)
        self.model.train()
        training_data = training_data.to(self.device)
        
        for epoch in range(epochs):
            total_loss = 0
            # np.random.shuffle(training_data)  # 打乱数据            
            
            for i in range(0, len(training_data), batch_size):
                batch_data = training_data[i:i+batch_size]#.to(self.device)
                
                if self.model_type == 'skipgram':
                    center_words = batch_data[:, 0]
                    context_words = batch_data[:, 1]
                    
                    # 负采样
                    negative_samples = self.trainer.get_negative_samples(
                        len(batch_data), context_words).to(self.device)
                    
                    # 前向传播
                    positive_score, negative_score = self.model(
                        center_words, context_words, negative_samples)
                    
                    # 计算损失
                    loss = self.trainer.negative_sampling_loss(positive_score, negative_score)
                
                else:  # CBOW
                    context_words = torch.tensor([d[0] for d in batch_data])
                    center_words = torch.tensor([d[1] for d in batch_data])
                    
                    # CBOW使用标准softmax损失（简化版）
                    scores = self.model(context_words)
                    loss = nn.CrossEntropyLoss()(scores, center_words)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(training_data) / batch_size)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    def get_word_vectors(self):
        """获取训练好的词向量"""
        return self.model.get_word_vectors()
    
    def find_similar_words(self, word, top_k=5):
        """查找相似词"""
        if word not in self.word2idx:
            return f"单词 '{word}' 不在词汇表中"
        
        word_idx = self.word2idx[word]
        word_vector = self.model.center_embeddings.weight.data[word_idx]
        
        # 计算与所有词的余弦相似度
        all_vectors = self.model.center_embeddings.weight.data[1:]
        similarities = torch.cosine_similarity(
            word_vector.unsqueeze(0), all_vectors, dim=1
        )
        
        # 获取最相似的词（排除自身和<UNK>）
        top_indices = similarities.argsort(descending=True)[:top_k] # [1:top_k+1]
        
        similar_words = []
        for idx in top_indices:
            similar_words.append((self.idx2word[idx.item()], similarities[idx].item()))
        
        return similar_words
    
    def word2vec(self, word):
        """将单词转换为向量"""
        if word not in self.word2idx:
            return None
        index =  self.word2idx[word]
        return self.model.get_word_vectors()[index]
    
    def vec2word(self, vector):
        """将向量转换为单词"""
        similarities = torch.cosine_similarity(
            torch.tensor(vector).unsqueeze(0), 
            self.model.center_embeddings.weight.data, dim=1
        )
        top_index = similarities.argmax().item()
        return self.idx2word[top_index]

# %%

def read_text(text):  #@save
    """将数据集加载到文本行的列表中"""
    with open(text, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return [line for line in lines]

# %%
text = 'rt-polarity.pos'
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
    window_size=2,
    learning_rate=0.08,
    device='cuda',
)
skipgram_trainer.train(epochs=50, batch_size=32)
etime = time.time()
print(f'训练完成，总耗时: {etime - stime:.2f} 秒； \n 平均耗时: {(etime - stime) / 50:.2f} 秒')

# %%
skipgram_trainer.word2idx[:6], skipgram_trainer.word_freq[:6]

# %%
skipgram_trainer.vec2word(skipgram_trainer.get_word_vectors()[:6])


