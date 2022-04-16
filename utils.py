import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

def get_reviews_for_product(product_name, product_review_map):
    return list(product_review_map[product_review_map['name'] == product_name]['reviews_text'])


def pre_process_text(review_text):
    stop = stopwords.words('english')
    review_text = ' '.join([word for word in review_text.split(
    ) if word not in (stop)])  # stop words removal
    review_text = review_text.lower()  # changing text to lower case
    review_text = review_text.translate(str.maketrans(
        '', '', string.punctuation))  # removing punctuation
    # removing digits and other characters
    review_text = re.sub("(\\W|\\d)", " ", review_text)
    review_text = lemmatize_text(review_text)  # Lemmatization
    return review_text


def lemmatize_text(review_text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(review_text)
    lemmatized_output = ' '.join(
        [lemmatizer.lemmatize(token) for token in tokens])
    return lemmatized_output