# coding=utf-8
import pandas
import math
from sklearn.metrics import roc_auc_score

import sys
sys.path.append("..")
from shad_util import print_answer

# 1. Загрузите данные из файла data-logistic.csv. Это двумерная выборка, целевая переменная на которой
# принимает значения -1 или 1.

df = pandas.read_csv('data-logistic.csv', header=None)
y = df[0]
X = df.loc[:, 1:]

# 2. Убедитесь, что выше выписаны правильные формулы для градиентного спуска. Обратите внимание, что мы используем
# полноценный градиентный спуск, а не его стохастический вариант!


def fw1(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in xrange(0, l):
        S += y[i] * X[1][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w1 + (k * (1.0 / l) * S) - k * C * w1


def fw2(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in xrange(0, l):
        S += y[i] * X[2][i] * (1.0 - 1.0 / (1.0 + math.exp(-y[i] * (w1*X[1][i] + w2*X[2][i]))))

    return w2 + (k * (1.0 / l) * S) - k * C * w2


# 3. Реализуйте градиентный спуск для обычной и L2-регуляризованной (с коэффициентом регуляризации 10)
# логистической регрессии. Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).

def grad(y, X, C=0.0, w1=0.0, w2=0.0, k=0.1, err=1e-5):
    i = 0
    i_max = 10000
    w1_new, w2_new = w1, w2

    while True:
        i += 1
        w1_new, w2_new = fw1(w1, w2, y, X, k, C), fw2(w1, w2, y, X, k, C)
        e = math.sqrt((w1_new - w1) ** 2 + (w2_new - w2) ** 2)

        if i >= i_max or e <= err:
            break
        else:
            w1, w2 = w1_new, w2_new

    return [w1_new, w2_new]

# 4. Запустите градиентный спуск и доведите до сходимости (евклидово расстояние между векторами весов на соседних
# итерациях должно быть не больше 1e-5). Рекомендуется ограничить сверху число итераций десятью тысячами.

w1, w2 = grad(y, X)
rw1, rw2 = grad(y, X, 10.0)

# 5. Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании?
# Эти величины будут ответом на задание. Обратите внимание, что на вход функции roc_auc_score нужно подавать
# оценки вероятностей, подсчитанные обученным алгоритмом. Для этого воспользуйтесь сигмоидной функцией:
# a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)).


def a(X, w1, w2):
    return 1.0 / (1.0 + math.exp(-w1 * X[1] - w2 * X[2]))


y_score = X.apply(lambda x: a(x, w1, w2), axis=1)
y_rscore = X.apply(lambda x: a(x, rw1, rw2), axis=1)

auc = roc_auc_score(y, y_score)
rauc = roc_auc_score(y, y_rscore)

print_answer(1, "{:0.3f} {:0.3f}".format(auc, rauc))

