# CountVectorizer로 클래스 만들기
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()

# 각 단어의 빈도 수 출력
print('bag of words vector : ', vector.fit_transform(corpus).toarray())

# 각 단어의 인덱스가 어찌 부여되었는 지 출력
print('vocabulary : ', vector.vocabulary_)