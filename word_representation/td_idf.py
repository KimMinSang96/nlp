import pandas as pd     # 데이터 프레임을 위해 사용됨
from math import log    # log 계산을 위해 사용됨

docs = ["really delicious banana", "really delicious apple", "yellow banana banana", "I like fruits"]

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

# docs의 수
N = len(docs)

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N / (df + 1))

def tfidf(t, d):
    return tf(t, d)* idf(t)


result = []

for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = vocab)
tf_

result = []

for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
idf_

result = []

for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d))

tfidf_ = pd.DataFrame(result, columns = vocab)
print(tfidf_)
