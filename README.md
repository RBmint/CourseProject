#### CS410 Final Project, Fall 2021
#### Royce Zhai (mzhai4@illinois.edu)

### Introduction

The topic is about sentiment analysis on tweets from Twitter. 
The task is to pass tweets to the model and predict the sentiment in the tweets.
It is interesting because we can identify whether people are being positive or negative in their posts.
The approach is to train models on a certain dataset containing tweets.

### Tools & libraries used

- `Python 3.7`
- `Jupyter` - notebooks used to train and test the models
- `Pickle` - used to save the trained models and vectors as binary files
- `Pandas, NumPy` - load and manipulate data using DataFrames
- `NLTK` - used in data pre-processing and cleaning
- `Scikit-learn` - machine learning algorithm toolkit
- `Tweepy` - Twitter API to stream live tweets

### Environment setup using conda

Make sure you have [Anaconda](https://www.anaconda.com/distribution/) installed.

Create the environment:  `conda create -n rbmint python=3.7`

Activate the environment:  `conda activate rbmint`

Install packages: `pip install -r environment_setup.txt`

Install stopwords:  `python -c "import nltk; nltk.download('stopwords')"`

### Files in this workspace

- `app.py` - Main application file that interacts with the tweets and the models
- `TrainModel.ipynb` - This notebook contains the pre-processing and model training
- `environment_setup.txt` - File containing the Python requirements for this project
- `Test/` directory 
    - `Test.ipynb` - Notebook containing test code to unpack and load the model for predictions
    - `twitter_analysis.py` - Initial tests using the Twitter API and the trained models
    - `twitter_api.py` - Initial tests setting up the Twitter API
    - `TweetStreamAnalysis.txt` - Test file containing tweets saved after running a stream
- `Pickled data/` directory
    - `LR.pickle` - Pickled trained Logistic regression model
    - `naive-bayes.pickle` - Pickled trained Naive Bayes model
    - `nn.pickle` - Pickled trained Neural Network model
    - `vector.pickle` - Pickled TF-IDF vector to transform the data

### Data pre-processing steps

- 1.6 m individual tweets with a 1 (Positive) or 0 (Negative) label
- Data cleaning involved the following steps
    - Convert the tweet to lowercase, remove stopwords
    - Remove the hashtag symbol (`#`)
    - Remove `@` mentions, websites
    - Perform stemming

### Models and accuracy achieved

- `Logistic Regression` - 77%
- `Naive Bayes` - 76%
- `Neural Network` - 71%

### How to use the app

- `app.py` is a command line app that supports the following arguments
    - Tweets from a specific user
        - `--user` or `-u` - username of the user to fetch tweets from (example - taylorswift13 (without the `@`))
        - `--count` or `-c` - number of tweets to fetch and analyze (example - 5, defaults to 10)
    - Stream tweets for a list of topics
        - `--stream` - list of topics to fetch live tweets from Twitter and perform analysis (example - "illinois" "football")
        - `--time` or `-t` - total duration of the stream in seconds (example - 10, defaults to 5)
        - `--file` - save the tweets and performed analysis to a file named `TweetStreamAnalysis.txt` in the current workspace
        - `--visualize` or `-v` - visualizes the predictions using a pie chart. Saves to file when `--file` flag is used

### Examples

#### Using Streams with multiple topics for a period of 10 seconds
```
❯ python app.py --stream "illinois" "football" --time 10                                                                                                                                                                ─╯
```

#### Picking a specific user and fetching last 10 tweets
```
❯ python app.py --user taylorswift13 --count 10                                                                                                                                                                                                  ─╯
```

#### Streaming topics and visualizing the results
```
❯ python app.py --stream "pokemon" "winter" --time 10 --visualize                                                                                                                                                                             ─╯
```

### Software Usage tutorial
Please see the link below:
