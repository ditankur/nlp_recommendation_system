from flask import Flask, render_template, url_for, request
from constants import *
from utils import *
import pickle
import pandas as pd

# Load recommendation system
rec_system = pickle.load(open(RECOMMENDATION_ENGINE, 'rb'))

# Load vectorizer
tfidf_vectorizer = pickle.load(open(VECTORIZER, 'rb'))

# Load sentiment analysis model
sentiment_model = pickle.load(open(SENTIMENT_ANALYSIS_MODEL, 'rb'))

# Load product name and review text mapping
product_review_map = pickle.load(open(PRODUCT_NAME_AND_REVIEW_TEXT_MAP, 'rb'))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def submit():
    user_name = request.form['username']
    recommended_products = get_recommended_products(user_name)
    return render_template('results.html', products=recommended_products)


def get_recommended_products(user_name):

    # Top 20 products from recommendation system
    ratings = rec_system.loc[user_name].sort_values(ascending=False)[0:20]
    products = pd.concat([ratings], axis=1).index.values.tolist()

    product_prediction_dict = dict()

    for product_name in products:

        # Get the product reviews
        product_reviews = get_reviews_for_product(
            product_name, product_review_map)

        # Get reviews after preprocessing
        processed_reviews_text = list(map(pre_process_text, product_reviews))

        # Transform reviews data using vectorizer
        transformed_reviews = tfidf_vectorizer.transform(
            processed_reviews_text)

        # Predict sentiment
        predictions = list(sentiment_model.predict(transformed_reviews))

        pos_predictions_perc = (sum(predictions)/len(predictions)) * 100

        # Create dict having product name as KEY and percentage of positive sentiment as VALUE
        product_prediction_dict[product_name] = pos_predictions_perc

    top_five_recommendations = {k: product_prediction_dict[k] for k in sorted(
        product_prediction_dict, key=product_prediction_dict.get, reverse=True)[:5]}

    return top_five_recommendations



if __name__ == '__main__':
    app.run()
