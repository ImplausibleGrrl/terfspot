# christina lu
# build_features.py

import pandas as pd

data_path = '../../../summer/'

# FEATURE TYPE: FOLLOWING -----------------------------------------------------
# given list of 2000 most followed users in following_vector_file
# create feature vector of size 2000, 1 if followed and 0 if not
following_vector_file = './base_following_vector.csv'
following_data_file = data_path + 'following_negative.json'

# gets base following vector from file
def get_top_vector(filename):
    df = pd.read_csv(filename)

    return df[df.columns[0]].tolist()

# build following features and return df
def build_following_features(filename):
    vec = get_top_vector(following_vector_file)

    data_vecs = []
    # iterate through json file containing raw following data
    with open(filename, 'r') as json_file:
        obj = ''
        append = False

        for line in json_file:
            # indicates we have one json object
            if '}' in line:
                obj += '}'
                append = False
                user = json.loads(obj)
                obj = ''

                user_vec = [int(user['user_id'])]
                friends = set(user['friend_ids'])

                for id in vec:
                    if id in friends:
                        user_vec.append(1)
                    else:
                        user_vec.append(0)

                data_vecs.append(user_vec)

            if append:
                obj += line

            if '{' in line:
                append = True
                obj += '{'

    column_names = 'user_id' + vec
    df = pd.DataFrame(data_vecs, columns=column_names)

    return df

# FEATURE TYPE: SIGNAL TWEET --------------------------------------------------
tweet_file = data_path + 'tweets_negative.csv'
topic_model = './topic_model/topics_25.model'
# select top signal tweet for each user according to topic model

# apply topic model to all tweets
# add column to df of topic 12 probability
def apply_topic_model():
    pass

# select top tweet under topic 12 for each user
# return df of user_id and top tweet text
def select_top_tweet():
    pass

# bert encode

def build_tweet_features():
    pass
