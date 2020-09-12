import argparse
import os
import os.path
import logging
import numpy as np
import pandas as pd
import clean_parse_json as cleaner
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SpatialDropout1D, Conv1D, MaxPooling1D, Activation, Embedding, Flatten, \
    GlobalMaxPooling1D, LSTM
from keras.preprocessing.text import Tokenizer
from keras import regularizers, callbacks, optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import pickle
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


pan_dir = ''
model_name = 'glove_model_a0_w0'
target_column = "target"
timeline_column = "timeline"
text_column = "text"
glove_file_name = 'glove.840B.300d.txt'
# glove_file_name = 'glove.twitter.27B.200d.txt'


def evaluate_model(model, x_test, y_test):

    y_predict = (np.asarray(model.predict(x_test))).round()

    acc = metrics.accuracy_score(y_test, y_predict)
    logging.info('Accuracy: {}'.format(acc))

    conf_matrix = metrics.confusion_matrix(y_test, y_predict)
    logging.info('Confusion matrix: {}'.format(conf_matrix))

    precision = metrics.precision_score(y_test, y_predict)
    logging.info('Precision score: {}'.format(precision))

    recall = metrics.recall_score(y_test, y_predict)
    logging.info('Recall score: {}'.format(recall))

    val_f1 = metrics.f1_score(y_test, y_predict)
    logging.info('F1 score: {}'.format(val_f1))

    val_auc = metrics.roc_auc_score(y_test, y_predict)
    logging.info('Auc score: {}'.format(val_auc))

    model_plot_file = os.path.join(pan_dir, 'models', 'glove', '{}.png'.format(model_name))
    plot_model(model, to_file=model_plot_file, show_shapes=True, show_layer_names=True)


def get_text(dict):
    if timeline_column not in dict:
        return
    timeline = dict[timeline_column]
    text = ""
    for tweet in timeline:
        if tweet["lang"] != "en":
            continue
        text += " " + tweet[text_column]
    # text = cleaner.clean_text(text)
    text = cleaner.process_text(text)
    return text


def get_data(file_path):
    print('Loading {} data'.format(file_path))
    df = pd.read_json(file_path)
    df = df["data"]
    print('preprocessing data')
    x_train = df.apply(get_text)
    y_train = df.apply(lambda dict: 1 if dict[target_column].strip() == "bot" else 0)
    return x_train, y_train


def load_data():
    train_df_file = os.path.join(pan_dir, 'data', 'train.json')
    test_df_file = os.path.join(pan_dir, 'data', 'test.json')
    train_text, train_target = get_data(train_df_file)
    test_text, test_target = get_data(test_df_file)
    return train_text, train_target, test_text, test_target


def create_embedding_weights_matrix(word_vectors, word_index, embedding_dims=300):
    weights_matrix = np.zeros((len(word_index) + 1, embedding_dims))

    count = 0
    for word, idx in word_index.items():
        if word in word_vectors:
            weights_matrix[idx] = word_vectors[word]
            count += 1
    logging.info('Words found on word2vec: {}'.format(count))

    return weights_matrix


def load_embedding_layer(tokenizer, seq_len):
    vocab_size = len(tokenizer.word_index) + 1
    logging.info('Vocab size: {}'.format(vocab_size))

    # Load word vectors
    logging.info("Loading Glove word2vec vectors")
    model = load_glove_model()
    weights_matrix = create_embedding_weights_matrix(model.wv, tokenizer.word_index)

    return Embedding(input_dim=vocab_size,
                     output_dim=weights_matrix.shape[1],
                     input_length=seq_len,
                     weights=[weights_matrix],
                     trainable=False
                     )


def define_conv_model(tokenizer, seq_len, filters=64, kernel_size=4, hidden_dims=256):
    model = Sequential()

    print('seq_len: {}'.format(seq_len))
    embedding_layer = load_embedding_layer(tokenizer, seq_len=seq_len)

    model.add(embedding_layer)
    model.add(Dropout(0.5))

    model.add(Conv1D(filters,
                     kernel_size,
                     activation='relu'))
    model.add(Dropout(0.5))
    # model.add(SpatialDropout1D(0.5))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(filters,
                     kernel_size,
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=4))

    model.add(GlobalMaxPooling1D())
    # model.add(Flatten())

    model.add(Dense(hidden_dims,
                    activation='relu'
                    ))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model


def define_deep_conv_model(tokenizer, seq_len, filters=64, kernel_size=4, hidden_dims=256):
    model = Sequential()

    vocab_size = len(tokenizer.word_index) + 1
    print('seq_len: {}'.format(seq_len))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=300, input_length=seq_len)
    model.add(embedding_layer)
    model.add(Dropout(0.5))

    for i in range(0, 5):
        model.add(Conv1D(filters,
                         kernel_size,
                         activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))

    # model.add(GlobalMaxPooling1D())
    model.add(Flatten())

    for i in range(0, 2):
        model.add(Dense(hidden_dims,
                        activation='relu'
                        ))
        model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model


def train_model(model, x_train, y_train, x_test, y_test, batch_size, learning_rate):
    model_dir = os.path.join(pan_dir, 'data', 'models', 'glove')
    model_location = os.path.join(model_dir, '{}.h5'.format(model_name))
    model_weights_location = os.path.join(model_dir, '{}_weights.h5'.format(model_name))

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


def load_pretrained_model(model_name):
    model_file = os.path.join(pan_dir, 'data', 'models', 'glove', "{}.h5".format(model_name))
    model = load_model(model_file)
    return model


def print_classes(dataset_type, y_data):
    humans = 0
    bots = 0
    for clazz in y_data:
        if clazz == 1:
            bots += 1
        else:
            humans += 1
    print('dataset type: {}'.format(dataset_type))
    print('bots length: {}'.format(bots))
    print('humans length: {}'.format(humans))


def load_tokenizer(x_train, num_words=None):
    file_path = os.path.join(pan_dir, 'data', 'tokenizers', 'tokenizer_{}.pickle'.format(num_words))
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
        logging.info('Tokenizer loaded from disk')
        return tokenizer
    else:
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(x_train)

        with open(file_path, 'wb') as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('Tokenizer fit on texts and stored on disk')

        return tokenizer


def load_glove_model():
    print("loading embedding from " + glove_file_name)
    glove_file = os.path.join(pan_dir, 'data', 'glove', glove_file_name)
    word2vec_file = os.path.join(pan_dir, 'data', 'glove', glove_file_name.replace('txt','word2vect'))
    if not os.path.exists(word2vec_file):
        _ = glove2word2vec(glove_file, word2vec_file)
    glove_model = KeyedVectors.load_word2vec_format(word2vec_file)
    return glove_model


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

    global pan_dir
    pan_dir = args.path

    logs_path = os.path.join(pan_dir, 'logs', 'glove_model_log.log')
    logging.basicConfig(filename=logs_path, filemode='w',
                        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    evaluate_mode = args.evaluate
    learning_rate = float(args.learning_rate)
    batch_size = 32
    model_name = args.model
    x_train, y_train, x_test, y_test = load_data()
    print('Data loaded successfully')
    lengths = np.array([len(text) for text in x_train])
    print('Count: {}'.format(len(lengths)))
    print('Min length: {}'.format(lengths.min()))
    print('Avg length: {}'.format(lengths.mean()))
    print('Std length: {}'.format(lengths.std()))
    print('Max length: {}'.format(lengths.max()))
    print('Count of sequences > 11000: {}'.format(len([length for length in lengths if length > 11000])))
    print_classes("train", y_train)
    print_classes("test", y_test)

    seq_length = int(lengths.mean() + lengths.std())

    tokenizer = load_tokenizer(x_train)

    train_sequences = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(train_sequences, maxlen=seq_length, padding='post')

    val_sequences = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(val_sequences, maxlen=seq_length, padding='post')

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
