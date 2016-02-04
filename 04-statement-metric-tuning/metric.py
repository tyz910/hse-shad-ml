# coding=utf-8
import pandas
import sklearn
from numpy import linspace
from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

import sys
sys.path.append("..")
from shad_util import print_answer

# 1. Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston(). Результатом вызова данной функции
# является объект, у которого признаки записаны в поле data, а целевой вектор — в поле target.

data = load_boston()
X = data.data
y = data.target

# 2. Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.

X = sklearn.preprocessing.scale(X)

# 3. Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, чтобы всего было протестировано
# 200 вариантов (используйте функцию numpy.linspace). Используйте KNeighborsRegressor с n_neighbors=5 и
# weights='distance' — данный параметр добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей.
# В качестве метрики качества используйте среднеквадратичную ошибку (параметр scoring='mean_squared_error' у
# cross_val_score). Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с
# random_state = 42, не забудьте включить перемешивание выборки (shuffle=True).


def test_accuracy(kf, X, y):
    scores = list()
    p_range = linspace(1, 10, 200)
    for p in p_range:
        model = KNeighborsRegressor(p=p, n_neighbors=5, weights='distance')
        scores.append(cross_val_score(model, X, y, cv=kf, scoring='mean_squared_error'))

    return pandas.DataFrame(scores, p_range).max(axis=1).sort_values(ascending=False)


kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)
accuracy = test_accuracy(kf, X, y)

# 4. Определите, при каком p качество на кросс-валидации оказалось оптимальным (обратите внимание,
# что показатели качества, которые подсчитывает cross_val_score, необходимо максимизировать).
# Это значение параметра и будет ответом на задачу.

top_accuracy = accuracy.head(1)
print_answer(1, top_accuracy.index[0])
