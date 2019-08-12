import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_extraction import calculate_features, calculate_features_by_specific_values
from utils import add_suffix_to_file_name, fix_dataset, plot_confision_matrix, LABEL_NAME


class FakeTweetsDetector:

    def __init__(self,clusters_n=5):
        self.clusters_n=clusters_n
        self.dimensionality_reducer = TruncatedSVD(n_components=2)

    def fit(self,X,y):
        # reduce dimensionality for k-means
        self.dimensionality_reducer.fit(X)
        X_reduced = self.dimensionality_reducer.transform(X)
        self.kmeans = KMeans(n_clusters=self.clusters_n).fit(X_reduced)
        # compute the maximal distance of a point to the centroid in each cluster - this is the threshold for really assigning a point to a cluster
        X_distances = pd.DataFrame(self.kmeans.transform(X_reduced))
        X_distances['cluster'] = self.kmeans.labels_
        self.max_distances = [X_distances[X_distances['cluster']==c][c].max() for c in range(self.kmeans.n_clusters)]
        self.mnb = MultinomialNB(alpha=0.01).fit(X,y)

    def predict(self,X):
        predictions = pd.DataFrame(columns=['prediction'],index=range(len(X)))
        # first - detect fake tweets
        # reduce dimensionality
        X_reduced = self.dimensionality_reducer.transform(X)
        # compute distances from centroids
        X_distances = pd.DataFrame(data=self.kmeans.transform(X_reduced))
        X_distances['pred_cluster'] = self.kmeans.predict(X_reduced)
        # a tweet really belongs to a cluster if its distance from the centroid is smaller than the farthest point in each cluster
        X_distances['in_cluster'] = X_distances.apply(
            lambda x: x[int(x['pred_cluster'])] <= self.max_distances[int(x['pred_cluster'])], axis=1)
        fake_index = X_distances[X_distances['in_cluster']==False].index
        predictions.loc[fake_index,'prediction'] = 'Fake'

        # second - for the not fake tweets, we predict hillary or trump by the NB model
        legit_index = X_distances[X_distances['in_cluster']==True].index
        legit_tweets = X.iloc[legit_index]
        nb_preds = self.mnb.predict(legit_tweets)
        predictions.loc[legit_index,'prediction'] = nb_preds
        return predictions['prediction']



if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--testPerc', action='store',type=float, default=0.3, help='Size of the testset in evaluation (default=0.3)')
    argParser.add_argument('dataset_path', help='path to the dataset file')
    argParser.add_argument('--testSet', default=None, action='store', help='path to the testset file')
    args = argParser.parse_args()

    # The dataset might contain tweets with commas in the tweet text - which disrupts the csv format
    fixed_file_name = add_suffix_to_file_name(os.path.basename(args.dataset_path), '_fixed')
    fixed_dataset_path = os.path.join(os.path.dirname(args.dataset_path), fixed_file_name)
    fix_dataset(args.dataset_path, fixed_dataset_path)

    # Load dataset
    dataset = pd.read_csv(fixed_dataset_path)

    # Calculate features
    features = calculate_features(dataset)

    if args.testSet != None:   # testset is given
        # fix it and load it
        fixed_test_path = add_suffix_to_file_name(os.path.basename(args.testSet), '_fixed')
        fix_dataset(args.testSet, fixed_test_path)
        testset = pd.read_csv(fixed_test_path)
        test_features = calculate_features_by_specific_values(testset,features.columns)
        train_set, train_labels = features, dataset[LABEL_NAME]
        test_set, test_labels = test_features, testset[LABEL_NAME]
    else:
        # split to train and test
        train_set, test_set, train_labels, test_labels = train_test_split(features, dataset[LABEL_NAME], test_size=args.testPerc)

    # build model
    fakeDetector = FakeTweetsDetector(clusters_n=5)
    fakeDetector.fit(train_set,train_labels)
    preds = fakeDetector.predict(test_set)

    # evaluate results
    from sklearn.metrics import classification_report, confusion_matrix

    classes = sorted(preds.unique())
    if 'Fake' not in classes:
        classes.insert(0,'Fake')
    print(classification_report(test_labels, preds))
    conf_m = confusion_matrix(test_labels, preds)
    if conf_m.shape == (2,2):   # no fakes were found, add zeros to represent it in the confusion matrix
        conf_m = np.array([[0,0,0],[0]+conf_m[0].tolist(),[0]+conf_m[1].tolist()])
    plot_confision_matrix(conf_m,classes)