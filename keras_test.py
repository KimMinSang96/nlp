from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import nltk

nltk.download('stopwords')

# corpus를 받음
raw_text = "It’s been a problem for the ages but Cameroonian soccer great Samuel Eto’o appears determined to stamp it out. Cameroon sealed its qualification for the 2023 Africa Cup of Nations’ Under-17s tournament with a 2-0 win against the Republic of Congo 2-0 on January 15, but the squad which won that match was unrecognizable from the one initially selected for the event. That’s because 21 players of the 30-man original group were disqualified for failing age eligibility tests following MRI scans to determine bone age and then ejected from the team, according to BBC Sport. To make matters worse, 11 of the replacements drafted into the squad also failed tests and were too old to play in the qualifiers. The ejection of those players followed Cameroon Football Association (FECAFOOT) President Eto’o’s decision to test players ahead of the competition."

# sentence 단위로 tokenize
sentences = sent_tokenize(raw_text)

print(sentences)

# preprocess init
vocab = {}
preprocessed_sentences = []
# 불용어를 영어로 설정
stop_words = set(stopwords.words('english'))

# preprocess 진행 -> 단어 길이가 2보다 큰 경우만 추출
for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence:
        word = word.lower()
        # stop_words : 불용어 -> 분석에 큰 의미가 없는 단어들 ex) i, me, my, myself...
        if word not in stop_words:
            if len(word) > 2:
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    
    preprocessed_sentences.append(result)

print(preprocessed_sentences)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(preprocessed_sentences)

print("tokenizer.word_index : ", tokenizer.word_index)

print("tokenizer.word_counts : ", tokenizer.word_counts)

# 숫자로 옮김
print('tokenizer.texts_to_sequences : ', tokenizer.texts_to_sequences(preprocessed_sentences))

