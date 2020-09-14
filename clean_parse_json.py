import argparse
import pandas as pd
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import math

USER_COLUMN = "userId"
TARGET_COLUMN = "target"
TIMELINE_COLUMN = "timeline"
TEXT_COLUMN = "text"


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def process_text(text):
    if isfloat(text):
        try:
            if math.isnan(text):
                return ''
        except TypeError:
            print('text: {}'.format(text))
            return ''

    # remove links from the text
    text = re.sub(r'https?:\/\/.*[\r\n]*', 'https ', text, flags=re.MULTILINE)
    # split into words
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return ' '.join(words).strip()

def get_text(tweet):
    if tweet["lang"] != "en":
        return ""
    return tweet[TEXT_COLUMN]


def process_data(file_path):
    print('Loading {} data'.format(file_path))
    data_frame = pd.read_json(file_path)
    print('preprocessing data')
    data_frame = pd.concat([pd.DataFrame(x) for x in data_frame['data']])
    data_frame[TARGET_COLUMN] = data_frame[TARGET_COLUMN].apply(lambda target: 1 if target.strip() == "bot" else 0)
    data_frame[TIMELINE_COLUMN] = data_frame[TIMELINE_COLUMN].apply(get_text)
    data_frame = data_frame.groupby([USER_COLUMN, TARGET_COLUMN])[TIMELINE_COLUMN].apply(' '.join).reset_index()
    data_frame[TIMELINE_COLUMN] = data_frame[TIMELINE_COLUMN].apply(process_text)
    print('storing data')
    data_frame.to_pickle(file_path.replace('json', 'pkl'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", '-p', default=".",
                        help="Use this argument to change the training data directory')")
    args = parser.parse_args()
    data_path = args.path
    print("Data path: {}".format(data_path))
    dataset_file_path = os.path.join(data_path, 'data', 'dataset.json')
    if os.path.isfile(dataset_file_path):
        process_data(dataset_file_path)
    else:
        print("file {} does not exits".format(dataset_file_path))

if __name__ == "__main__":
    main()