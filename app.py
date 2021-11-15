import argparse
import pickle
import os

### Pickled models
base_pickle_dir = '' 
models = ['naive-bayes.pickle'] 
models_path = [*map(lambda file : os.path.join(base_pickle_dir, file), models)]

class Model:

    '''
    This class creates objects for the pre-trained models.
    '''

    ## TF-IDF vector required to transform the tweet
    # Vector specifications: Max features - 10,000, Ngram range - (1, 2)
    vector = None

    def __init__(self, model = None):

        self.models = dict(zip(['NaiveBayes'], models_path))

        
        self.model = self.import_model(self.models.get(model))

        if not Model.vector:
            Model.vector = self.init_vector()

    def import_model(self, model_path):
        '''
        Loads the corresponding model from the pickle.
        '''

        with open('naive-bayes.pickle', 'rb') as md:
            return pickle.load(md)

    def init_vector(self):
        '''
        Load the trained TF-IDF vector from pickle.
        '''

        with open('vector.pickle', 'rb') as vc:
            return pickle.load(vc)

    def label_prediction(self, prediction):
        '''
        Converts integer predictions to string.
        '''

        return 'Positive' if prediction[-1] == 1 else 'Negative'


class Classifier:

    '''
    Creates an object that loads the model to perform analysis.
    '''

    def __init__(self):
        self.NB = Model('NaiveBayes')
        self.models = {'NaiveBayes': self.NB}

    def weighted_average(self, data):
        '''
        The prediction of the Classifier is weighted.
        '''
        weights = {'NaiveBayes': 1} 
        
        total = {'Positive' : 0, 'Negative': 0}

        for model, score in data.items():
            total[score] += weights[model]

        return (max(total, key = total.get), total[max(total, key = total.get)])

    def predict(self, text, generate_summary = True):
        '''
        Predicts the sentiment of a tweet using all the imported models.
        '''

        predictions = dict()
        if isinstance(text, str):
            for name, model in self.models.items():
                predictions[name] = model.predict(text)
            return self.get_summary(predictions) if generate_summary else predictions

    def get_summary(self, predictions):
        '''
        Using the raw predictions, generates a weighted pretty-printed summary.
        '''
        result = str()

        for name, score in predictions.items():
            result += '{}: {}\n'.format(name, score)

        final_score = self.weighted_average(predictions)
        result += 'Prediction: {} with a probability of {}%\n'.format(final_score[0], final_score[-1]*100)
        return result

   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Tweet sentiment analyzer')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--user', '-u', type=str, default=None, help='Twitter username to fetch tweets')
    args = parser.parse_args()

    if args.user:
        # Initialize classifier
        model = Classifier()
        model.process_data(data = tweets, user = args.user, save_to_file = args.file, visualize = args.visualize)

