# coding=utf-8
import pandas
import re

import sys
sys.path.append("..")
from shad_util import print_answer


data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data['Pclass'] = data['Pclass'].astype(object)

# 1. Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.

sex_counts = data['Sex'].value_counts()
print_answer(1, '{} {}'.format(sex_counts['male'], sex_counts['female']))

# 2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).

surv_counts = data['Survived'].value_counts()
surv_percent = 100.0 * surv_counts[1] / surv_counts.sum()
print_answer(2, "{:0.2f}".format(surv_percent))

# 3. Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен).

pclass_counts = data['Pclass'].value_counts()
pclass_percent = 100.0 * pclass_counts[1] / pclass_counts.sum()
print_answer(3, "{:0.2f}".format(pclass_percent))

# 4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.

ages = data['Age'].dropna()
print_answer(4, "{:0.2f} {:0.2f}".format(ages.mean(), ages.median()))

# 5. Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.

corr = data['SibSp'].corr(data['Parch'])
print_answer(5, "{:0.2f}".format(corr))

# 6. Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name)
# его личное имя (First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию. Попробуйте вручную разобрать
# несколько значений столбца Name и выработать правило для извлечения имен, а также разделения их на женские и мужские.


def clean_name(name):
    # Первое слово до запятой - фамилия
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)

    # Если есть скобки - то имя пассажира в них
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)

    # Удаляем обращения
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)

    # Берем первое оставшееся слово и удаляем кавычки
    name = name.split(' ')[0].replace('"', '')

    return name


names = data[data['Sex'] == 'female']['Name'].map(clean_name)
name_counts = names.value_counts()
print_answer(6, name_counts.head(1).index.values[0])
