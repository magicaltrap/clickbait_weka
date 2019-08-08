#stopwords
from nltk.corpus import stopwords

#subjectivity, polarity
from textblob import TextBlob


def count_words(word_list):
    return len(word_list)


def average_word_length(word_list):
    length = 0
    for word in word_list:
        length += len(word)

    return length / len(word_list)

def longest_word(word_list):
    return len(max(word_list, key=len))


def start_with_number(word_list):
    if word_list[0].isdigit():
        return True
    else:
        return False


def start_with_question_word(word_list):
    question_words = ['who', 'what', 'why', 'where', 'when', 'how']

    if word_list[0] in question_words:
        return True
    else:
        return False



def number_of_stop_words(word_list):
    stop_words = set(stopwords.words('english'))

    count_stop_words = 0
    for token in word_list:
        if token in stop_words:
            count_stop_words += 1

    return count_stop_words



def sentiment_polarity_subjectivity(word_list):
    # check if parameter is a list. If yes, join the elements to a string
    if isinstance(word_list, list):
        text = " ".join(word_list)

    blob = TextBlob(text)
    polarity, subjectivity = blob.sentiment

    return polarity, subjectivity

def number_of_capital_letters(word_list):

    if isinstance(word_list, list):
        text = " ".join(word_list)

    return sum(1 for char in text if char.isupper())



def number_of_allcaps_words(word_list):

    number_allcap = 0

    for token in word_list:
        if token.isupper():
            number_allcap += 1

    return number_allcap


def number_of_punctuation(word_list):

    number_of_punctuation = 0
    for token in word_list:
        for char in token:
            if char in ['!', '?']:
                number_of_punctuation += 1

    return number_of_punctuation


def extract_bait_words(word_list):
    bait_word_list = ['you', 'best', 'greatest', 'weirdest', 'most', 'worst', 'funniest', 'incredible', 'secret',
                      'remarkable', 'miracle', 'magic', 'easier', 'should', 'popular', 'never', 'need', 'world',
                      'happened', 'guy', 'ever', 'cheapest', 'grossest']

    number_of_bait_words = 0
    presence_of_bait_words = False

    for token in word_list:
        if token in bait_word_list:
            number_of_bait_words += 1
            presence_of_bait_words = True

    return number_of_bait_words, presence_of_bait_words



# string = "Hallo you das ist ein string!!!!!"
# string1 = string.strip().split()
# print(extract_bait_words(string1))
#
# sequence = ["MORE", "NUMBERS", "are", "always", "good.", "best", "easier"]
# sequence = [x.lower() for x in sequence]
# print(extract_bait_words(sequence))
#
#
# print(int(999 * 0.25))