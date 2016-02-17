# coding=utf-8
import pandas
import sklearn.metrics as metrics

import sys
sys.path.append("..")
from shad_util import print_answer

# 1. Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки (колонка true) и ответы
# некоторого классификатора (колонка predicted).

df = pandas.read_csv('classification.csv')

# 2. Заполните таблицу ошибок классификации. Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям.
# Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1.
# Ответ в данном вопросе — четыре числа через пробел.

clf_table = {'tp': (1, 1), 'fp': (0, 1), 'fn': (1, 0), 'tn': (0, 0)}
for name, res in clf_table.iteritems():
    clf_table[name] = len(df[(df['true'] == res[0]) & (df['pred'] == res[1])])

print_answer(1, '{tp} {fp} {fn} {tn}'.format(**clf_table))

# 3. Посчитайте основные метрики качества классификатора:

# Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
acc = metrics.accuracy_score(df['true'], df['pred'])

# Precision (точность) — sklearn.metrics.precision_score
pr = metrics.precision_score(df['true'], df['pred'])

# Recall (полнота) — sklearn.metrics.recall_score
rec = metrics.recall_score(df['true'], df['pred'])

# F-мера — sklearn.metrics.f1_score
f1 = metrics.f1_score(df['true'], df['pred'])

# В качестве ответа укажите эти четыре числа через пробел.
print_answer(2, '{:0.2f} {:0.2f} {:0.2f} {:0.2f}'.format(acc, pr, rec, f1))

# 4. Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и значения степени
# принадлежности положительному классу для каждого классификатора на некоторой выборке:
# для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# для SVM — отступ от разделяющей поверхности (колонка score_svm),
# для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.

df2 = pandas.read_csv('scores.csv')

# 5. Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор имеет наибольшее значение
# метрики AUC-ROC (укажите название столбца с ответами этого классификатора)?
# Воспользуйтесь функцией sklearn.metrics.roc_auc_score.

scores = {}
for clf in df2.columns[1:]:
    scores[clf] = metrics.roc_auc_score(df2['true'], df2[clf])

print_answer(3, pandas.Series(scores).sort_values(ascending=False).head(1).index[0])

# 6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
# Какое значение точности при этом получается? Чтобы получить ответ на этот вопрос, найдите все точки
# precision-recall-кривой с помощью функции sklearn.metrics.precision_recall_curve. Она возвращает три массива:
# precision, recall, thresholds. В них записаны точность и полнота при определенных порогах,указанных в массиве
# thresholds. Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.

pr_scores = {}
for clf in df2.columns[1:]:
    pr_curve = metrics.precision_recall_curve(df2['true'], df2[clf])
    pr_curve_df = pandas.DataFrame({'precision': pr_curve[0], 'recall': pr_curve[1]})
    pr_scores[clf] = pr_curve_df[pr_curve_df['recall'] >= 0.7]['precision'].max()

print_answer(4, pandas.Series(pr_scores).sort_values(ascending=False).head(1).index[0])
