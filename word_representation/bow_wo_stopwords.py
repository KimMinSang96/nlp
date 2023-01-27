# CountVectorizer로 클래스 만들기
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

text = ["Family is not an important thing. It's everything."]
stop_words = set(stopwords.words('english'))

# Stop_word를 기존의 stop_word로 등록해 사용 
vector = CountVectorizer(stop_words = stopwords.words('english'))

# 각 단어의 빈도 수 출력
print('bag of words vector : ', vector.fit_transform(text).toarray())

# 각 단어의 인덱스가 어찌 부여되었는 지 출력
print('vocabulary : ', vector.vocabulary_)