import pandas as pd     # 데이터 프레임을 위해 사용됨
from math import log    # log 계산을 위해 사용됨
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'you know I wnat your love',
    'I like you',
    'what should I do'
]

vector = CountVectorizer()

# corpus로부터 각 단어의 빈도수 기록
print(vector.fit_transform(corpus).toarray())

# 단어와 맵핑된 인덱스 출력
print(vector.vocabulary_)

tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)

