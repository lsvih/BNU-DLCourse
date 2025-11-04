"""
NNML
Author: Yanzeng
直接用spam.csv来训一个神经网络语言模型并尝试predict next token
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader

df = pd.read_csv('spam.csv', encoding='latin-1')
texts = df['v2'].astype(str).tolist()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text

cleaned_texts = [clean_text(t) for t in texts]
all_tokens = []
for text in cleaned_texts:
    tokens = text.split()
    all_tokens.extend(['unk'] + tokens)

vocab = sorted(set(all_tokens))
vocab_size = len(vocab)
word2idx = {w: i + 1 for i, w in enumerate(vocab)}
idx2word = {i + 1: w for i, w in enumerate(vocab)}
idx2word[0] = 'unk'

print(f"Vocabulary size: {vocab_size}")
print(f"Sample tokens: {all_tokens[:10]}")
print(f"Sample vocab: {list(word2idx.items())[:5]}")


# 设定历史窗口大小（用几个词预测下一个）
context_size = 5

data = []
for tokens in [all_tokens]:
    tokens = tokens[:10000]
    for i in range(len(tokens) - context_size):
        context = tokens[i:i + context_size]
        target = tokens[i + context_size]
        context_idxs = [word2idx.get(w, 0) for w in context]  # 未知词默认为 0
        target_idx = word2idx.get(target, 0)
        data.append((context_idxs, target_idx))


print(f"Number of training samples: {len(data)}")

class NextTokenDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context_idxs, target_idx = self.data[idx]
        return torch.tensor(context_idxs, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)

dataset = NextTokenDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class NNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(NNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        total_embedding_dim = context_size * embedding_dim

        self.fc1 = nn.Linear(total_embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, context_size)
        embeds = self.embeddings(x)  # (batch_size, context_size, embedding_dim)
        embeds = embeds.view(embeds.size(0), -1)  # (batch_size, context_size * emb_dim)
        out = self.fc1(embeds)
        out = self.relu(out)
        out = self.fc2(out)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs

embedding_dim = 50
hidden_dim = 128

model = NNLM(vocab_size=vocab_size + 1,
             embedding_dim=embedding_dim,
             context_size=context_size,
             hidden_dim=hidden_dim)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)

num_epochs = 20

for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        context, target = batch
        optimizer.zero_grad()
        log_probs = model(context)  # [batch_size, vocab_size]
        loss = criterion(log_probs, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def get_top_k_predictions(model, context_words, word2idx, idx2word, k=5):
    model.eval()
    with torch.no_grad():
        context = context_words[-context_size:]
        context_idxs = [word2idx.get(w, 0) for w in context]
        if len(context_idxs) < context_size:
        	context_idxs = [0] * (context_size - len(context_idxs)) + context_idxs
        context_tensor = torch.tensor([context_idxs], dtype=torch.long)  # [1, context_size]

        log_probs = model(context_tensor)  # [1, vocab_size]
        probs = torch.exp(log_probs)       # logits to prob

        probs_np = probs.squeeze(0).cpu().numpy()
        topk_probs, topk_indices = torch.topk(probs, k=k)

        topk_probs = topk_probs.cpu().numpy()[0]
        topk_indices = topk_indices.cpu().numpy()[0]

        predictions = []
        for i in range(k):
            token = idx2word[topk_indices[i]]
            prob = topk_probs[i]
            predictions.append((token, prob))

        return predictions

def interactive_prediction_loop(model, word2idx, idx2word, context_size=2, k=5):
    print("\nNNLM")
    print(f"规则：输入上下文（如 'i love'），将预测接下来最可能的词")
    print("输入 'quit'、'q' 或 'exit' 退出程序")
    print("输入 'RESET'（大写）清空历史，重新从第一个词开始\n")

    previous_context_words = []

    while True:
        if previous_context_words:
            prompt = f">>> 当前上下文：{' '.join(previous_context_words)} "
        else:
            prompt = ">>> 请输入上下文（用空格分隔，或 RESET 重置）: "

        user_input = input(prompt).strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if user_input.upper() == 'RESET':
            previous_context_words = []
            print("已重置，现在可以输入新的起始词。")
            continue

        if not user_input:
            print("输入不能为空")
            continue

        user_input_clean = clean_text(user_input)
        context_words = user_input_clean.split()

        previous_context_words += context_words

        try:
            top_predictions = get_top_k_predictions(
                model, previous_context_words, word2idx, idx2word, k=k
            )

            print("\n最可能出现的词是：")
            for i, (token, prob) in enumerate(top_predictions, 1):
                print(f"  {i}. '{token}' \t (Prob: {prob:.4f})")

        except Exception as e:
            print(e)

        print()

interactive_prediction_loop(model, word2idx, idx2word, context_size=2, k=5)