import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

values = [["spam_1", 1], ["none_1", 0],["spam_2", 1], ["none_2", 0], ["none_3", 0]]

columns = ["mails", "spams"]

df = pd.DataFrame(values, columns = columns)

x = df['mails']
y = df['spams']

# Data Frame 표현
print("df : ", df)

# X데이터 Y데이터를 표현하고 있음
print("x data : ", x.to_list())
print("y data : ", y.to_list())

# numpy 이용하여 분리하기
np_array = np.arange(0, 16).reshape((4, 4))

print("np array : ")
print(np_array)

x = np_array[:, :3]
y = np_array[:, 3]

print("X data : ")
print(x)
print("Y data : ", y)

# 매우 중요
# X -> 독립 변수 데이터, Y -> 종속 변수 데이터, test_size -> 테스트용 데이터 개수 지정, 1보다 작을 경우 비율 나타냄
# train_size -> 트레이닝용 데이터 개수 지정, 1보다 작을 경우 비율을 나타냄
# random_state -> 난수 시드
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1234)
print('x training data : ')
print(x_train)
print('x test data : ')
print(x_test)