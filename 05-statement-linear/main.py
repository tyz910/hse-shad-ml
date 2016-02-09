# coding=utf-8
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("..")
from shad_util import print_answer

# 1. Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
# Целевая переменная записана в первом столбце, признаки — во втором и третьем.

df_train = pandas.read_csv('perceptron-train.csv', header=None)
y_train = df_train[0]
X_train = df_train.loc[:, 1:]

df_test = pandas.read_csv('perceptron-test.csv', header=None)
y_test = df_train[0]
X_test = df_train.loc[:, 1:]

# 2. Обучите персептрон со стандартными параметрами и random_state=241.

model = Perceptron(random_state=241)
model.fit(X_train, y_train)

# 3. Подсчитайте качество (долю правильно классифицированных объектов, accuracy) полученного классификатора
# на тестовой выборке.

acc_before = accuracy_score(y_test, model.predict(X_test))

# 4. Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Обучите персептрон на новых выборках. Найдите долю правильных ответов на тестовой выборке.

model = Perceptron(random_state=241)
model.fit(X_train_scaled, y_train)
acc_after = accuracy_score(y_test, model.predict(X_test_scaled))

# 6. Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.
# Это число и будет ответом на задание.

print_answer(1, acc_after - acc_before)
