# coding=utf-8
import pandas
from sklearn.decomposition import PCA
from numpy import corrcoef

import sys
sys.path.append("..")
from shad_util import print_answer


# 1. Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый
# день периода.

df = pandas.read_csv('close_prices.csv')
X = df.loc[:, 'AXP':]

# 2. На загруженных данных обучите преобразование PCA с числом компоненты равным 10. Скольких компонент хватит,
# чтобы объяснить 90% дисперсии?

pca = PCA(n_components=10)
pca.fit(X.values)

var = 0
n_var = 0
for v in pca.explained_variance_ratio_:
    n_var += 1
    var += v
    if var >= 0.9:
        break

print_answer(1, n_var)

# 3. Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.

df_comp = pandas.DataFrame(pca.transform(X))
comp0 = df_comp[0]

# 4. Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv. Чему равна корреляция Пирсона между первой
# компонентой и индексом Доу-Джонса?

df2 = pandas.read_csv('djia_index.csv')
dji = df2['^DJI']
corr = corrcoef(comp0, dji)
print_answer(2, corr[1, 0])

# 5. Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.

comp0_w = pandas.Series(pca.components_[0])
comp0_w_top = comp0_w.sort_values(ascending=False).head(1).index[0]
company = X.columns[comp0_w_top]
print_answer(3, company)
