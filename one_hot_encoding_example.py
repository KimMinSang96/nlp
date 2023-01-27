from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

text = "Hello, my name is kim. Nice to meet you."

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print("word set : ", tokenizer.word_index)

sub_text = "Hello, Nice to meet you."
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)

one_hot = to_categorical(encoded)

print(one_hot)