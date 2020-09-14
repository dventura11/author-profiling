import argparse
import os
import logging
import clean_parse_json as cleaner
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras import callbacks, optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import pickle

file_path = ''
MODEL_NAME = 'dl_model'
TARGET_COLUMN = "target"
TIMELINE_COLUMN = "timeline"


def load_data(dataset_name):
    data_frame_file = os.path.join(file_path, 'data', dataset_name + '.pkl')
    if not os.path.exists(data_frame_file):
        dataset_file_path = os.path.join(file_path, 'data', dataset_name + '.json')
        cleaner.process_data(dataset_file_path)
    data_frame = pd.read_pickle(data_frame_file)
    data_frame = data_frame[[TIMELINE_COLUMN, TARGET_COLUMN]]
    print_classes('full', data_frame[TARGET_COLUMN])
    train_df, test_df = train_test_split(data_frame, test_size=0.3, random_state=1092, shuffle=True)
    return train_df[TIMELINE_COLUMN], train_df[TARGET_COLUMN], test_df[TIMELINE_COLUMN], test_df[TARGET_COLUMN]


def load_tokenizer(x_train, num_words=None):
    tokenizer_file_path = os.path.join(file_path, 'data', 'tokenizers')
    if not os.path.exists(tokenizer_file_path):
        os.mkdir(tokenizer_file_path)
    tokenizer_file_path = os.path.join(tokenizer_file_path, 'tokenizer_{}.pickle'.format(num_words))
    if os.path.isfile(tokenizer_file_path):
        with open(tokenizer_file_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
        logging.info('Tokenizer loaded from disk')
        return tokenizer
    else:
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(x_train)

        with open(tokenizer_file_path, 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Tokenizer fit on texts and stored on disk')

        return tokenizer


def load_pretrained_model(model_name):
    model_file = os.path.join(file_path, 'data', 'models', "{}.h5".format(model_name))
    model = load_model(model_file)
    return model


def define_conv_model(tokenizer, seq_len, filters=64, kernel_size=4, hidden_dims=256):
    model = Sequential()

    vocab_size = len(tokenizer.word_index) + 1
    print('seq_len: {}'.format(seq_len))
    print('vocab_size: {}'.format(vocab_size))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, input_length=seq_len)

    model.add(embedding_layer)
    model.add(Dropout(0.5))

    model.add(Conv1D(filters,
                     kernel_size,
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(filters,
                     kernel_size,
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=4))

    model.add(GlobalMaxPooling1D())

    model.add(Dense(hidden_dims,
                    activation='relu'
                    ))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model


def train_model(model, x_train, y_train, x_test, y_test, batch_size, learning_rate):
    model_dir = os.path.join(file_path, 'data', 'models')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_location = os.path.join(model_dir, '{}.h5'.format(MODEL_NAME))
    model_weights_location = os.path.join(model_dir, '{}_weights.h5'.format(MODEL_NAME))

    # Implement Early Stopping
    early_stopping_callback = callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=0,
                                                      patience=5,
                                                      verbose=1)
    #   restore_best_weights=True)
    save_best_model = callbacks.ModelCheckpoint(model_weights_location, monitor='val_loss', verbose=1,
                                                save_best_only=True, mode='auto')

    adam = optimizers.Adam(lr=learning_rate, decay=0.01)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=1000,
                        verbose=2,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping_callback, save_best_model])

    # reload best weights
    model.load_weights(model_weights_location)

    logging.info('Model trained. Storing model on disk.')
    model.save(model_location)
    logging.info('Model stored on disk.')


def evaluate_model(model, x_test, y_test):
    y_predict = (np.asarray(model.predict(x_test))).round()

    acc = metrics.accuracy_score(y_test, y_predict)
    logging.info('Accuracy: {}'.format(acc))
    print_classes('expected', y_test)
    print_classes('predict', y_predict)
    conf_matrix = metrics.confusion_matrix(y_test, y_predict)
    logging.info('Confusion matrix:\n {}'.format(conf_matrix))

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_predict).ravel()
    print('tp: {}\nfp: {}\ntn: {}\nfn: {}'.format(tp, fp, tn, fn))

    precision = metrics.precision_score(y_test, y_predict)
    logging.info('Precision score: {}'.format(precision))

    recall = metrics.recall_score(y_test, y_predict)
    logging.info('Recall score: {}'.format(recall))

    val_f1 = metrics.f1_score(y_test, y_predict)
    logging.info('F1 score: {}'.format(val_f1))

    val_auc = metrics.roc_auc_score(y_test, y_predict)
    logging.info('Auc score: {}'.format(val_auc))

    model_plot_file = os.path.join(file_path, 'data', 'models', '{}.png'.format(MODEL_NAME))
    plot_model(model, to_file=model_plot_file, show_shapes=True, show_layer_names=True)


def print_classes(dataset_type, y_data):
    humans = 0
    bots = 0
    for clazz in y_data:
        if clazz == 1:
            bots += 1
        else:
            humans += 1
    print('----------------------------------------')
    print('dataset type: {}'.format(dataset_type))
    print('dataset size: {}'.format(len(y_data)))
    print('bots length: {}'.format(bots))
    print('humans length: {}'.format(humans))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", '-p', default=".",
                        help="Use this argument to change the training data directory')")
    parser.add_argument("--model", '-m', default="",
                        help="Use this argument to continue training a stored model")
    parser.add_argument("--learning_rate", '-l', default="0.001",
                        help="Use this argument to set the learning rate to use. Default: 0.001")
    parser.add_argument("--evaluate", '-e', action='store_true', default="False",
                        help="Use this argument to set run on evaluation mode")
    args = parser.parse_args()

    global file_path
    file_path = args.path
    logs_path = os.path.join(file_path, 'logs')
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    logs_path = os.path.join(logs_path, MODEL_NAME + '.log')
    logging.basicConfig(filename=logs_path, filemode='w',
                        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    evaluate_mode = args.evaluate
    learning_rate = float(args.learning_rate)
    batch_size = 32
    model_name = args.model
    x_train, y_train, x_test, y_test = load_data('dataset')
    lengths = np.array([len(text) for text in x_train])
    print('Count: {}'.format(len(lengths)))
    print('Data loaded successfully')
    print_classes("train", y_train)
    print_classes("test", y_test)
    print('Min length: {}'.format(lengths.min()))
    print('Avg length: {}'.format(lengths.mean()))
    print('Std length: {}'.format(lengths.std()))
    print('Max length: {}'.format(lengths.max()))
    print('Count of sequences > 11000: {}'.format(len([length for length in lengths if length > 11000])))

    seq_length = int(lengths.mean() + lengths.std())

    tokenizer = load_tokenizer(x_train)

    train_sequences = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(train_sequences, maxlen=seq_length, padding='post')

    test_seq = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(test_seq, maxlen=seq_length, padding='post')

    if model_name:
        model = load_pretrained_model(model_name)
    else:
        model = define_conv_model(tokenizer, seq_length)

    logging.info(model.summary())

    if evaluate_mode is True:
        evaluate_model(model, x_test, y_test)
    else:
        train_model(model, x_train, y_train, x_test, y_test, batch_size, learning_rate)
        evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
