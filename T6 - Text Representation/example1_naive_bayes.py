import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict


def load_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    labels = df.iloc[:, 0].values
    emails = df.iloc[:, 1].values
    return labels, emails

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return words


def build_vocabulary_and_counts(labels, emails):
    """
    构建词汇表，并统计每个类别（spam/ham）中每个词出现的次数
    返回：
    - vocabulary：所有单词的集合（去重）
    - spam_word_counts：spam 类别中每个词的出现次数
    - ham_word_counts：ham 类别中每个词的出现次数
    - spam_total_words：spam 类别总词数
    - ham_total_words：ham 类别总词数
    - spam_count：spam 邮件总数
    - ham_count：ham 邮件总数
    """
    spam_word_counts = defaultdict(int)
    ham_word_counts = defaultdict(int)
    spam_total_words = 0
    ham_total_words = 0
    spam_count = 0
    ham_count = 0

    vocabulary = set()

    for label, email in zip(labels, emails):
        words = preprocess_text(email)
        for word in words:
            vocabulary.add(word)
            if label == 'spam':
                spam_word_counts[word] += 1
                spam_total_words += 1
                spam_count += 1
            elif label == 'ham':
                ham_word_counts[word] += 1
                ham_total_words += 1
                ham_count += 1

    return (vocabulary, spam_word_counts, ham_word_counts,
            spam_total_words, ham_total_words, spam_count, ham_count)


class NaiveBayesSpamClassifier:
    def __init__(self):
        self.vocab = None
        self.spam_word_counts = None
        self.ham_word_counts = None
        self.spam_total_words = None
        self.ham_total_words = None
        self.spam_count = None
        self.ham_count = None
        self.p_spam = None
        self.p_ham = None
        self.alpha = 1  # 拉普拉斯平滑 (Laplace smoothing)

    def train(self, labels, emails):
        (self.vocab, self.spam_word_counts, self.ham_word_counts,
         self.spam_total_words, self.ham_total_words,
         self.spam_count, self.ham_count) = build_vocabulary_and_counts(labels, emails)

        total_emails = self.spam_count + self.ham_count
        self.p_spam = self.spam_count / total_emails
        self.p_ham = self.ham_count / total_emails

    def predict(self, email):
        words = preprocess_text(email)
        log_p_spam = np.log(self.p_spam)
        log_p_ham = np.log(self.p_ham)

        vocab_size = len(self.vocab)

        for word in words:
            # P(word|spam) with Laplace smoothing
            count_word_spam = self.spam_word_counts.get(word, 0)
            p_word_given_spam = (count_word_spam + self.alpha) / (self.spam_total_words + self.alpha * vocab_size)

            # P(word|ham) with Laplace smoothing
            count_word_ham = self.ham_word_counts.get(word, 0)
            p_word_given_ham = (count_word_ham + self.alpha) / (self.ham_total_words + self.alpha * vocab_size)

            log_p_spam += np.log(p_word_given_spam)
            log_p_ham += np.log(p_word_given_ham)

        # 比较 logP(spam|email) 和 logP(ham|email)
        if log_p_spam > log_p_ham:
            return 'spam'
        else:
            return 'ham'

    def evaluate(self, labels, emails):
        correct = 0
        for label, email in zip(labels, emails):
            pred = self.predict(email)
            if pred == label:
                correct += 1
        accuracy = correct / len(labels)
        return accuracy


def plot_top_keywords_with_matplotlib(spam_word_counts, ham_word_counts, vocabulary, top_n=10):
    spam_total = sum(spam_word_counts.values())
    ham_total = sum(ham_word_counts.values())

    keywords_data = []

    for word in vocabulary:
        cnt_spam = spam_word_counts.get(word, 0)
        cnt_ham = ham_word_counts.get(word, 0)

        # 加一平滑
        p_spam = (cnt_spam + 1) / (spam_total + 2)
        p_ham = (cnt_ham + 1) / (ham_total + 2)

        if p_ham == 0:
            log_odds = float('inf')
        elif p_spam == 0:
            log_odds = float('-inf')
        else:
            log_odds = np.log(p_spam / p_ham)

        keywords_data.append((word, cnt_spam, cnt_ham, log_odds))

    # 按 log_odds 降序排序（spam 倾向性强的词）
    keywords_data.sort(key=lambda x: x[3], reverse=True)
    top_keywords = keywords_data[:top_n]

    words = [x[0] for x in top_keywords]
    spam_counts = [x[1] for x in top_keywords]
    ham_counts = [x[2] for x in top_keywords]
    log_odds = [x[3] for x in top_keywords]

    x = np.arange(len(words))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width/2, spam_counts, width, label='Spam', color='crimson', alpha=0.8)
    rects2 = ax.bar(x + width/2, ham_counts, width, label='Ham', color='skyblue', alpha=0.8)

    ax.set_xlabel('Keywords', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Top {top_n} Keywords: Spam vs Ham (Based on Frequency)', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()


def main():
    filepath = 'spam.csv'
    labels, emails = load_data(filepath)

    # 20% for testing.
    split_idx = int(0.8 * len(labels))
    train_labels = labels[:split_idx]
    train_emails = emails[:split_idx]
    test_labels = labels[split_idx:]
    test_emails = emails[split_idx:]

    print("Training...")
    classifier = NaiveBayesSpamClassifier()
    classifier.train(train_labels, train_emails)

    accuracy = classifier.evaluate(test_labels, test_emails)
    print(f"ACC: {accuracy:.2f} ({accuracy*100:.1f}%)")

    sample_emails = ["Congratulations！You've won a $1000 Walmart gift card. Click here to claim now!",
                    "Congratulations on your work anniversary! Here's to many more years."]
    for sample_email in sample_emails:
        prediction = classifier.predict(sample_email)
        print(f"\n示例邮件: '{sample_email}'")
        print(f"预测结果: {prediction}")

    input()

    plot_top_keywords_with_matplotlib(
        classifier.spam_word_counts,
        classifier.ham_word_counts,
        classifier.vocab,
        top_n=20
    )

if __name__ == "__main__":
    main()
