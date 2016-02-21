# coding=utf-8
import math
import pandas
import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave
from sklearn.cluster import KMeans

import sys
sys.path.append("..")
from shad_util import print_answer

# 1. Загрузите картинку parrots.jpg. Преобразуйте изображение, приведя все значения в интервал от 0 до 1.
# Для этого можно воспользоваться функцией img_as_float из модуля skimage.

image = img_as_float(imread('parrots.jpg'))
w, h, d = image.shape

# 2. Создайте матрицу объекты-признаки: характеризуйте каждый пиксель тремя координатами - значениями интенсивности
# в пространстве RGB.

pixels = pandas.DataFrame(np.reshape(image, (w*h, d)), columns=['R', 'G', 'B'])

# 3. Запустите алгоритм K-Means с параметрами init='k-means++' и random_state=241. После выделения кластеров все
# пиксели, отнесенные в один кластер, попробуйте заполнить двумя способами: медианным и средним цветом по кластеру.


def cluster(pixels, n_clusters=8):
    print 'Clustering: ' + str(n_clusters)

    pixels = pixels.copy()
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    pixels['cluster'] = model.fit_predict(pixels)

    means = pixels.groupby('cluster').mean().values
    mean_pixels = [means[c] for c in pixels['cluster'].values]
    mean_image = np.reshape(mean_pixels, (w, h, d))
    imsave('images/mean/parrots_' + str(n_clusters) + '.jpg', mean_image)

    medians = pixels.groupby('cluster').median().values
    median_pixels = [medians[c] for c in pixels['cluster'].values]
    median_image = np.reshape(median_pixels, (w, h, d))
    imsave('images/median/parrots_' + str(n_clusters) + '.jpg', median_image)

    return mean_image, median_image


# 4. Измерьте качество получившейся сегментации с помощью метрики PSNR. Эту метрику нужно реализовать
# самостоятельно (см. определение).


def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return 10 * math.log10(float(1) / mse)


# 5. Найдите минимальное количество кластеров, при котором значение PSNR выше 20 (можно рассмотреть
# не более 20кластеров). Это число и будет ответом в данной задаче.

for n in xrange(1, 21):
    mean_image, median_image = cluster(pixels, n)
    psnr_mean, psnr_median = psnr(image, mean_image), psnr(image, median_image)
    print psnr_mean, psnr_median

    if psnr_mean > 20 or psnr_median > 20:
        print_answer(1, n)
        break
