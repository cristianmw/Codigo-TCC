import sys
import os
import pandas as pd
import datetime
import shutil
import time
import io
from   sklearn.model_selection       import train_test_split

import tensorflow                    as     tf
from   tensorflow                    import keras
from   tensorflow.keras              import losses
from   tensorflow.keras              import metrics
from   tensorflow.keras.callbacks    import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from   tensorflow.keras.initializers import GlorotNormal, GlorotUniform, HeNormal
from   tensorflow.keras.layers       import Dense, Dropout, Input
from   tensorflow.keras.models       import load_model, Sequential
from   tensorflow.keras.optimizers   import Adam


def model_l():
    initializer = HeNormal()
    optimizer = Adam(learning_rate=0.0001)

    model = tf.keras.Sequential([
        Input(shape=(9,)),
        Dense(500,activation='relu',kernel_initializer=initializer),
        Dense(400,activation='relu',kernel_initializer=initializer),
        Dense(500,activation='relu',kernel_initializer=initializer),
        Dense(1, activation='linear')
        ])

    model.compile(optimizer=optimizer,loss=losses.MAE,metrics=[metrics.MAPE])

    return model


def dataset_generator(filename, batch_size):
    csvfile = open(filename)
    reader = pd.read_csv(csvfile, chunksize=batch_size, header=None)
    while True:
        for chunk in reader:
            w = chunk.values
            x = w[:, :9]
            y = w[::, 9]
            yield x, y
        csvfile = open(filename)
        reader = pd.read_csv(csvfile, chunksize=batch_size, header=None)


# treina um modelo de dnn
def train_dnn(model, n_batch, dataset, n_epochs_max=2000, patience=100, dir_log=''):
    model_name = 'model' + '_' + str(n_batch) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = dir_log + model_name


    model_weights_file_name = dir_name + '_weights.hdf5'
    model_save_file_name = dir_name + '_model.h5'

    model = model()

    train_set = './' + dataset + '/train_set.csv'
    validation_set = './' + dataset + '/validation_set.csv'
    test_set = './' + dataset + '/test_set.csv'

    early_stop = EarlyStopping(
        monitor='val_mean_absolute_percentage_error',
        mode='min',
        verbose=1,
        patience=patience,
        restore_best_weights=True)

    checkpoint_min_loss = ModelCheckpoint(
        filepath=dir_name + '_checkpoint_{epoch}_{loss:.4f}.hdf5',
        monitor='mean_absolute_percentage_error',
        verbose=1,
        save_best_only=True,
        mode='auto')

    # tensorboard callback
    log_dir = dir_log + 'logs_tensorboard/' + model_name
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # adicionar tensorboard_callback na lista para usar tensorboard
    callbacks_list = [early_stop, checkpoint_min_loss]

    train_generator = dataset_generator(train_set, n_batch)
    validation_generator = dataset_generator(validation_set, n_batch)
    test_generator = dataset_generator(test_set, n_batch)

    map_size = 200 ** 2
    sampling = 0.1
    num_maps = 9
    dataset_size = ((map_size * sampling) * (map_size * sampling -1) * num_maps) / 2

    ntrain = (dataset_size * 0.7) // n_batch
    nval = (dataset_size * 0.15) // n_batch
    ntest = (dataset_size * 0.15) // n_batch

    history = model.fit(
        train_generator,
        steps_per_epoch=ntrain,
        epochs=n_epochs_max,
        callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=nval,
        verbose=2
    )

    model.save(model_save_file_name)
    model.save_weights(model_weights_file_name)

    #score = model.evaluate(X_test, y_test, verbose=0)
    score = model.evaluate(test_generator, verbose=0)

    return score, history


def dataset_read(filename, random_state=21, size_train=0.7, size_validation=0.5, size_test=0.5):
    # le arquivo csv, dados ja estão normalizados
    dataframe = pd.read_csv(filename, sep=',')

    # obtem numero de colunas, comencando em zero
    number_columns = dataframe.shape[1] - 1

    # transforma dataframe em um array 2D, RANDOM
    a = len(dataframe)
    # print(a)
    dataframe = dataframe.sample(n=a, random_state=59)

    dataset = dataframe.values

    # slice em duas dimensoes, separa entradas, duas dimensoes e saida, uma dimensao, sem entrada, considera a ultima coluna como saida
    X = dataset[:, :number_columns]
    Y = dataset[:, number_columns]

    # separa os conjuntos de treinamento, validacao e teste
    X_train, X_validation_and_test, y_train, y_validation_and_test = train_test_split(X, Y, train_size=size_train,
                                                                                      random_state=random_state)
    X_validation, X_test, y_validation, y_test = train_test_split(X_validation_and_test, y_validation_and_test,
                                                                  test_size=size_validation)

    return X_train, y_train, X_validation, y_validation, X_test, y_test


# grava o conteudo de data_io num csv
def grava(filename,data_io):
    # grava os resultados num csv para analise posterior
    with open(filename, 'a') as file:
        data_io.seek(0)
        shutil.copyfileobj(data_io, file)


def main():
    args = sys.argv

    corte_name = args[1]
    dataset_location = args[2]

    data_io_stats = io.StringIO()
    stats_file_name = 'resultados_' + corte_name + '.csv'

    print(tf.config.experimental.list_physical_devices('GPU'))

    models = [model_l]
    batch_size = 1024 * 16

    print('Training')
    try:
        for model in models:
            model().summary()

            t1_start = time.time()
            score, history = train_dnn(model, batch_size, dataset_location, dir_log=corte_name+'/')
            t1_stop = time.time()
            diff_time = t1_stop - t1_start

            data_io_stats.write("""%s,%s,%s,%s,%s\n""" % ('model',batch_size,history.history,diff_time,score))

            print(score)
    finally:
        # grava os resultados num csv
        grava(stats_file_name, data_io_stats)


if __name__ == '__main__':
    main()