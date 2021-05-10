import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

tweets = pd.read_json('random_tweets.json', lines=True)

# get median number of retweets for classifying a "viral" tweet
median_retweets = tweets['retweet_count'].median()  # 13.0

# create column is_viral and set it to 1 if retweet count is greater than the median and 0 if not.
tweets['is_viral'] = np.where(tweets['retweet_count'] > median_retweets, 1, 0)

# create features
tweets['tweet_length'] = tweets.apply(lambda tweet: len(tweet['text']), axis=1)
tweets['followers_count'] = tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
tweets['friends_count'] = tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)

# print(len(tweets))
# print(tweets.columns)
# print(tweets.loc[0])

labels = tweets['is_viral']
data = tweets[['tweet_length', 'followers_count', 'friends_count']]

# scale the data
data = scale(data)

# split data into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=1)

# find best value for n_neighbors
scores = []
for k in range(1, 201):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_data, train_labels)
    score = classifier.score(test_data, test_labels)
    scores.append(score)

# plot the best value for k, best value is ≈ 0.62 with k ≈ 45, need better features to increase score
plt.plot(range(1, 201), scores)
plt.xlabel("k value")
plt.ylabel("test score")
plt.show()
