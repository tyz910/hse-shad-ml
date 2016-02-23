# coding=utf-8
import os
import pandas

def save_clean_data(cleaner, X_train, y_train, X_test, name='simple'):
    path = './data/clean/' + name
    if not os.path.exists(path):
        os.makedirs(path)

    y_train.to_csv(path + '/y_train.csv')
    cleaner(X_train).to_csv(path + '/X_train.csv')
    cleaner(X_test).to_csv(path + '/X_test.csv')

def get_clean_data(cleaner_name='simple'):
	path = './data/clean/' + cleaner_name
	X_train = pandas.read_csv(path + '/X_train.csv', index_col='match_id')
	y_train = pandas.read_csv(path + '/y_train.csv', index_col='match_id')
	X_test = pandas.read_csv(path + '/X_test.csv', index_col='match_id')
	return X_train, y_train['radiant_win'], X_test

def kaggle_save(name, model, X_test):
	y_test = model.predict_proba(X_test)[:, 1]
	result = pandas.DataFrame({'radiant_win': y_test}, index=X_test.index)
	result.index.name = 'match_id'
	result.to_csv('./data/kaggle/{}.csv'.format(name))
