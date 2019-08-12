import re
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

LABEL_NAME = 'handle'

def clean_text(text, ignore=[]):
    '''
    Converts the given text to lower case and remove punctuations, digits, stopwords, and if passed - other words.
    :param text: str - the text to clean
    :param ignore: custom list of words to remove from the text
    :return: str - the cleaned text
    '''
    word_sep_punct = ['-','#','~']
    # remove punctuation
    c_text = "".join([" " if c in word_sep_punct else c for c in text])
    c_text = "".join([c for c in c_text if c not in string.punctuation])
    # remove numbers
    c_text = re.sub('[0-9]+', '', c_text)
    # split bu white-spaces
    c_text = re.split('\W+', c_text)
    # remove empty words and lower-case
    c_text = [w.lower() for w in c_text if w != '']
    # remove stop words and common words (if given) and stem them
    stopwords = nltk.corpus.stopwords.words('english')
    words_to_ignore = stopwords + ignore
    c_text = [w for w in c_text if w not in words_to_ignore]
    return c_text

def get_hour_from_date_sting(date_str):
    '''
    Extract the hour of day from a date string
    :param date_str: date string
    :return: str - the hour of the day
    '''
    date_time = date_str.split('T')
    h_m_s = date_time[1].split(':')
    return h_m_s[0]

def fix_dataset(original_path,new_path):
    '''
    Fix the problems with tweets that contain commas and are not surrounded with " "
    :param original_path: path to the dataset path
    :param new_path: path to save the fixed dataset path
    :return:
    '''
    with open(original_path,'r', encoding="utf8") as csv_f, open(new_path,'w', encoding="utf8") as fixed_f:
        # first - copy headers
        headers = csv_f.readline()
        fixed_f.write(headers)
        num_of_attributes = len(headers.split(','))
        tweet_cont = False
        for line in csv_f:
            splitted_line = line.split(',')
            if not tweet_cont:
                # new instance
                # copy the data
                new_line = []
                # first 2 columns remain the same ('id' and 'handle')
                new_line += splitted_line[:2]
                # unite comma-separated tweets
                text = []
                curr_index = 2
            # aggregate tweet text
            while curr_index < len(splitted_line) and splitted_line[curr_index] not in ['True','False']:
                text.append(splitted_line[curr_index])
                curr_index += 1
            if curr_index >= len(splitted_line):  # tweet with new line char
                curr_index = 0
                tweet_cont = True
                continue
            tweet_cont = False
            pre = '' if text[0][0] == '"' else '"'
            suf = '' if text[-1][-1] == '"' else '"'
            new_line.append(pre + ','.join(text) + suf)
            # all of the remaining arguments are aligned
            new_line += splitted_line[curr_index:]
            # write new line
            fixed_f.write(','.join(new_line))

def add_suffix_to_file_name(file_name,suff):
    '''
    Append a suffix to the given file name
    :param file_name: str - a file name
    :param suff: str - the suffix to append
    :return: str - the new file name
    '''
    filename_split = file_name.split('.')
    return '.'.join(filename_split[:-1]) + suff + '.' + filename_split[-1]

def average_results(reports,confusion_mtrices,classes):
    '''
    Average the measures (precision, recall, f1-score, confusion matrix) in the given results set.
    :param reports: list of measures as returned by sklearn.metrics.classification_report
    :param confusion_mtrices: list of confusion matrices as returned by sklearn.metrics.confusion_matrix
    :param classes: list of the possible class values the model can output
    :return: dictionary with the averaged classification measured and an np.ndarray - averaged confusion matrix
    '''
    avg_report = {}
    for c in classes:
        avg_report[c] = {'precision' : sum([r[c]['precision'] for r in reports]) / len(reports),
                      'recall' : sum([r[c]['recall'] for r in reports]) / len(reports),
                      'f1-score' : sum([r[c]['f1-score'] for r in reports]) / len(reports)}
    avg_conf_matrix = sum(confusion_mtrices) / len(confusion_mtrices)
    return avg_report,avg_conf_matrix

def plot_confision_matrix(conf_matrix,classes):
    '''
    Plot the given confusion matrix
    :param conf_matrix: np.ndarray - a confusion matric
    :param classes: list of the possible class values
    :return:
    '''
    fig, ax = plt.subplots()
    # rotate y labels
    plt.setp(ax.get_yticklabels(), ha="center", rotation_mode="anchor")
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Reds', xticklabels=classes, yticklabels=classes).set(
        xlabel='Predicted value', ylabel='Real value')
    plt.show()