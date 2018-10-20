import graphlab

def sentimentAnalysis():
        products =  graphlab.SFrame('amazon_baby.gl/')
        products['word_count'] = graphlab.text_analytics.count_words(products['review'])
        giraffe_reviews = products[products['name'] == 'Vulli Sophie the Giraffe Teether']
        products = products[products['rating'] != 3]
        products['sentiment'] = products['rating'] >=4
        train_data,test_data = products.random_split(.8, seed=0)
        train_data, validation_data = train_data.random_split(.8, seed = 0)
	sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                             target='sentiment',
                                                             features=['word_count'],
                                                             validation_set=validation_data)
        sentiment_model.evaluate(test_data, metric='roc_curve')
	#import pdb; pdb.set_trace()
        #sentiment_model.show(view='Evaluation')
	giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type='probability')
        giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
        print("Most negative review: ",giraffe_reviews[-1]['review'])


def main():
        sentimentAnalysis()

if __name__ == "__main__":
main()
