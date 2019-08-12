import argparse
import sys, os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from feature_extraction import calculate_features
from utils import fix_dataset, add_suffix_to_file_name, LABEL_NAME, average_results, \
    plot_confision_matrix

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('dataset_path', help='path to the dataset file')
    args = argParser.parse_args()

    # The dataset might contain tweets with commas in the tweet text - which disrupts the csv format
    fixed_file_name = add_suffix_to_file_name(os.path.basename(args.dataset_path),'_fixed')
    fixed_dataset_path = os.path.join(os.path.dirname(args.dataset_path),fixed_file_name)
    fix_dataset(args.dataset_path,fixed_dataset_path)

    # Load dataset
    dataset = pd.read_csv(fixed_dataset_path)

    # Calculate features
    features = calculate_features(dataset)
    # split to 4 folds
    skf = StratifiedKFold(n_splits=4)
    folds = list(skf.split(features,dataset[LABEL_NAME]))

    # Evaluate the model
    mnb = MultinomialNB(alpha=0.01)
    results = []
    for train_index,test_index in folds:
        fold_train_feats,fold_train_labels = features.iloc[train_index],dataset[LABEL_NAME].iloc[train_index]
        fold_test_feats,fold_test_labels = features.iloc[test_index],dataset[LABEL_NAME].iloc[test_index]
        # train the model foe this fold
        mnb.fit(fold_train_feats, fold_train_labels)
        preds = mnb.predict(fold_test_feats)
        report = classification_report(fold_test_labels, preds,output_dict=True)
        conf_matrix = confusion_matrix(fold_test_labels, preds)
        results.append((report, conf_matrix, preds))

    # Display measures
    classes = dataset[LABEL_NAME].unique()
    avg_report, avg_conf_matrix = average_results([r[0] for r in results],[r[1] for r in results],classes)
    for c in classes:
        print('Class %s:\t Precision: %.2f\t Recall: %.2f\t F1-score: %.2f'%(c,avg_report[c]['precision'],avg_report[c]['recall'],avg_report[c]['f1-score']))
    plot_confision_matrix(avg_conf_matrix,classes)