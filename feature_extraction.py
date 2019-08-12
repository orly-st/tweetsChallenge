import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from utils import get_hour_from_date_sting, clean_text


def get_vocabulary_from_feature_names(features):
    '''
    Extract the word-related feature names (bag of words and 2-grams) from the given feature names list
    :param features: list of strings
    :return: list with the word-related feature names
    '''
    return [w for w in features if (w != 'is_retweet' and
                                    not w.startswith('hour_') and
                                    not w.startswith('lang_') and
                                    not w.startswith('src_'))]

def calculate_features(dataset,vocabulary=None):
    '''
    Extract the following features for each records in the dataset:
        1. bag of words (that appeared at least twice) - counts
        2. 2-grams (that appeared at least twice) - counts
        3. is retweet - boolean
        4. The hour in day that the tweet was tweeted - binary feature for each hour in the day
        5. foreign language - binary feature for each language that is not English in the dataset
        5. tweet's source - binary feature for each source in the dataset
    :param dataset: DataFrame containing the raw data from twitter
    :param vocabulary: optional - list of terms. If passed the word count features will be extracted according to this list
    :return: DataFrame with the calculated features corresponding each instance in the dataset
    '''
    # 1 - bag of words and 2-grams
    dataset['clean_text'] = dataset['text'].apply(lambda x: ' '.join(clean_text(x)))
    countVectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2)
    if vocabulary == None:
        countVector = countVectorizer.fit_transform(dataset['clean_text'])
    else:
        countVectorizer.fit(vocabulary)
        countVector = countVectorizer.transform(dataset['clean_text'])
    word_counts = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())

    # add unused words from vocabulary is given
    if vocabulary != None:
        for w in [w for w in vocabulary if w not in word_counts.columns]:
            word_counts[w] = 0

    # 2 - tweet-related and context
    word_counts['is_retweet'] = dataset['is_retweet']

    tweets_hour = dataset['time'].apply(get_hour_from_date_sting)
    hours_of_day = ['0' + str(i) for i in range(10)] + [str(i) for i in range(10, 24)]
    for h in hours_of_day:
        word_counts['hour_' + h] = tweets_hour.apply(lambda x: x == h)

    forigen_langs = [l for l in dataset['lang'].unique() if l != 'en']
    for l in forigen_langs:
        word_counts['lang_' + l] = dataset['lang'].apply(lambda x: x == l)

    sources = dataset['source_url'].unique()
    for s in sources:
        word_counts['src_' + s] = dataset['source_url'].apply(lambda x: x == s)

    return word_counts

def calculate_features_by_specific_values(dataset,feature_names):
    '''
    Similar to the calculate_features function - except that in this version the features are extracted according to the given feature names list
    :param dataset: DataFrame containing the raw data from twitter
    :param feature_names: list with the names of the features to extract
    :return: DataFrame with the calculated features corresponding each instance in the dataset
    '''
    # 1 - bag of words and 2-grams
    dataset['clean_text'] = dataset['text'].apply(lambda x: ' '.join(clean_text(x)))
    countVectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2)
    vocabulary = get_vocabulary_from_feature_names(feature_names)
    countVectorizer.fit(vocabulary)
    countVector = countVectorizer.transform(dataset['clean_text'])
    word_counts = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())

    # add unused words from vocabulary is given
    missing_words = [w for w in vocabulary if w not in word_counts.columns]
    for w in missing_words:
        word_counts[w] = 0

    # 2 - tweet-related and context
    if 'is_retweet' in feature_names:
        word_counts['is_retweet'] = dataset['is_retweet']

    h_of_day_feats = [f for f in feature_names if f.startswith('hour_')]
    if len(h_of_day_feats) > 0:
        tweets_hour = dataset['time'].apply(get_hour_from_date_sting)
        # hours_of_day = ['0' + str(i) for i in range(10)] + [str(i) for i in range(10, 24)]
        for h_feat in h_of_day_feats:
            h = h_feat.split('_')[1]
            word_counts[h_feat] = tweets_hour.apply(lambda x: x == h)

    lang_feats = [f for f in feature_names if f.startswith('lang_')]
    if len(lang_feats) > 0:
        # forigen_langs = [l for l in dataset['lang'].unique() if l != 'en']
        for l_feat in lang_feats:
            l = l_feat.split('_')[1]
            word_counts[l_feat] = dataset['lang'].apply(lambda x: x == l)

    src_feats = [f for f in feature_names if f.startswith('src_')]
    if len(src_feats) > 0:
        # sources = dataset['source_url'].unique()
        for s_feat in src_feats:
            s = s_feat.split('_')[1]
            word_counts[s_feat] = dataset['source_url'].apply(lambda x: x == s)

    return word_counts