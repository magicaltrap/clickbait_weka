import json
import arff #https://pypi.python.org/pypi/liac-arff
import argparse
import re
import itertools

from extracting_methods import *


parser = argparse.ArgumentParser()

parser.add_argument("--path_training", type=str, help="path to clickbait training jsonl file")
parser.add_argument("--path_truth", type=str, help="path to clickbait truth jsonl file")
parser.add_argument("--output_path_training", type=str, help="output path of arff training file")
parser.add_argument("--output_path_test", type=str, help="output path of arff test file")
parser.add_argument("--size_test_set", type=float, default=0.3, help="size of the test set in regard to the whole dataset (e.g. 0.2, 0.3, 0.4 of whole dataset")
args = parser.parse_args()


def extract_features(text):

    # 1) keep word_list with punctuation and capital letters
    word_list_with_punctuation = text.strip().split(' ')

    # 2) keep word_list with capital letters and without punctuation
    #removes punctuation
    text_no_punctuation = re.sub(r'[^\w\s]', '', text)
    word_list_with_capital_letters = text_no_punctuation.strip().split(' ')

    # 3) word_list with no punctuation and no capital letters
    word_list_lowercase = [word.lower() for word in word_list_with_capital_letters]


    f1 = 0 # word count
    f2 = 0 # average word length
    f3 = 0 # length of the longest word
    f4 = False # whether sentence start with number ['True', 'False']
    f5 = False # whether start who/what/why/where/when/how ['True', 'False']
    f6 = 0  # number of stopwords
    f7 = 0  # polarity (use word_list with punctuation)
    f8 = 0  # subjectivity (use word_list with punctuation)
    f9 = 0  # number of capital letters #use non-lowercase version for this method
    f10 = 0  # number of all capital WORDS #use non-lowercase version for this method
    f11 = 0  # number of punctuation (?, !) #use word_list with punctuation
    f12 = 0  # number of bait words
    f13 = False  # presence bait words


    f1 = count_words(word_list_lowercase)
    f2 = average_word_length(word_list_lowercase)
    f3 = longest_word(word_list_lowercase)
    f4 = start_with_number(word_list_lowercase)
    f5 = start_with_question_word(word_list_lowercase)
    f6 = number_of_stop_words(word_list_lowercase)
    f7, f8 = sentiment_polarity_subjectivity(word_list_with_punctuation) #punctuation influences scores
    f9 = number_of_capital_letters(word_list_with_capital_letters)
    f10 = number_of_allcaps_words(word_list_with_capital_letters)
    f11 = number_of_punctuation(word_list_with_punctuation)
    f12, f13 = extract_bait_words(word_list_lowercase)

    #if postText (Twitter post) has no words, return False
    if f1 == 0:
        return False
    return (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13)


# creates a training and a test set
def split_dicts(dictionary):
    length_test_set = int(len(dictionary) * args.size_test_set)          # length of smaller dataset
    iterator = iter(dictionary.items())

    test_set = dict(itertools.islice(iterator, length_test_set))   # grab first n (length_test_set) items
    training_set = dict(iterator)                        # grab the rest

    return training_set, test_set


if __name__ == '__main__':

    # feature description for the Weka .arff file later
    feature_names = [("id", 'STRING'),
                ("word count", 'NUMERIC'),
                ("average word length", 'NUMERIC'),
                ("length of the longest word", 'NUMERIC'),
                ("whether start with number", ['True', 'False']),
                ("whether start with who/what/why/where/when/how", ['True', 'False']),
                ("number of stopwords", 'NUMERIC'),
                ("polarity", 'NUMERIC'),
                ("subjectivity", 'NUMERIC'),
                ("number of capital letters", 'NUMERIC'),
                ("number of all capital WORDS", 'NUMERIC'),
                ("number of punctuation (?, !)", 'NUMERIC'),
                ("number of bait words", 'NUMERIC'),
                ("presence of bait words", ['True', 'False']),
                ("label", ['no-clickbait', 'clickbait'])]

    id_features = {}
    with open(args.path_training, 'r') as reader:
        for line in reader:
            instance = json.loads(line)

            if len(instance['postText'][0]) > 0:  # if postText has text (['postText'][0] -> a string)
                temp_feat = extract_features(instance['postText'][0])

                #check if postText (Twitter post) has text
                if temp_feat != False:
                    feat = (instance['id'],)
                    feat += temp_feat
                    # The method setdefault() is similar to get(), but will set dict[key] = default if key is not already in dict.
                    id_features.setdefault(instance['id'], feat)


    #create training and test set
    training_set, test_set = split_dicts(id_features)


    #get "clickbait" or "no_clickbait" label
    id_labels = {}
    with open(args.path_truth, 'r') as reader:
        for line in reader:
            instance = json.loads(line)

            label = 'clickbait'
            if instance['truthClass'] == 'no-clickbait':
                label = 'no-clickbait'
            if instance['id'] in id_features:
                id_labels.setdefault(instance['id'], label)

    data_training = {}
    data_training.setdefault('attributes', feature_names)
    data_training.setdefault('description', '')
    data_training.setdefault('relation', 'clickbait_sample')
    data_training.setdefault('data', [])

    for instance_id in training_set: #iterate through all keys
        tmp = [_ for _ in training_set[instance_id]]
        tmp.append(str(id_labels[instance_id]))
        #one instance is one list element
        data_training['data'].append(tmp)


    with open(args.output_path_training + "/clickbait_training.arff", 'w') as writer:
        writer.write(arff.dumps(data_training))


    data_test = {}
    data_test.setdefault('attributes', feature_names)
    data_test.setdefault('description', '')
    data_test.setdefault('relation', 'clickbait_sample')
    data_test.setdefault('data', [])

    for instance_id in test_set:  # iterate through all keys
        tmp = [_ for _ in test_set[instance_id]]
        tmp.append(str(id_labels[instance_id]))
        # one instance is one list element
        data_test['data'].append(tmp)

    with open(args.output_path_test + "/clickbait_test.arff", 'w') as writer2:
        writer2.write(arff.dumps(data_test))