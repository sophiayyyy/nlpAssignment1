#############################################################
## ASSIGNMENT 1 CODE SKELETON
## RELEASED: 2/6/2019
## DUE: 2/15/2019
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import io


#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    ## YOUR CODE HERE...
    tp = fp = 0
    for index in range(len(y_pred)):
        if y_pred[index] == 1 and y_true[index] == 1:
            tp = tp + 1
        elif y_pred[index] == 1 and y_true[index] == 0:
            fp = fp + 1
        tp = float(tp)
        fp = float(fp)
    precision = tp / (tp + fp)
    return precision


## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    ## YOUR CODE HERE...
    tp = fn = 0
    for index in range(len(y_pred)):
        if y_pred[index] == 1 and y_true[index] == 1:
            tp = tp + 1
        elif y_pred[index] == 0 and y_true[index] == 1:
            fn = fn + 1
    tp = float(tp)
    fn = float(fn)
    recall = tp / (tp + fn)
    return recall


## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    ## YOUR CODE HERE...
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = 2 * (precision * recall) / (precision + recall)
    return fscore


#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []
    with io.open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words, labels = load_file(data_file)
    pred_labels = [1] * len(labels)
    precision = get_precision(pred_labels, labels)
    recall = get_recall(pred_labels, labels)
    fscore = get_fscore(pred_labels, labels)
    performance = [precision, recall, fscore]
    return performance

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file):
    ## YOUR CODE HERE
    training_words, training_labels = load_file(training_file)
    training_max = 0
    dev_max = 0
    training_threshold = 0
    dev_threshold = 0
    for i in range(1, 10):
        length_threshold = i
        training_pred_labels = [0] * len(training_words)
        for index in range(len(training_words)):
            if len(training_words[index]) >= length_threshold:
                training_pred_labels[index] = 1
            else:
                training_pred_labels[index] = 0
        tprecision = get_precision(training_pred_labels, training_labels)
        trecall = get_recall(training_pred_labels, training_labels)
        tfscore = get_fscore(training_pred_labels, training_labels)
        training_performance = [tprecision, trecall, tfscore]
        if tfscore > training_max:
            training_max = tfscore
            training_threshold = i
            final_training_performance = training_performance
        dev_words, dev_labels = load_file(development_file)
        dev_pred_labels = [0] * len(dev_words)
        for index in range(len(dev_words)):
            if len(dev_words[index]) >= length_threshold:
                dev_pred_labels[index] = 1
            else:
                dev_pred_labels[index] = 0
        dprecision = get_precision(dev_pred_labels, dev_labels)
        drecall = get_recall(dev_pred_labels, dev_labels)
        dfscore = get_fscore(dev_pred_labels, dev_labels)
        development_performance = [dprecision, drecall, dfscore]
        if tfscore > dev_max:
            dev_max = tfscore
            dev_threshold = i
            final_dev_performance = development_performance
    #print ("best_training_threshold:" , training_threshold , final_training_performance, "best_dev_threshold:", dev_threshold, final_dev_performance)
    return final_training_performance, final_dev_performance


### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file):
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

def word_frequency_threshold(training_file, development_file, counts):
    ## YOUR CODE HERE
    training_max = 0
    dev_max = 0
    training_threshold = 0
    dev_threshold = 0
    final_training_performance = 0
    training_words, training_labels = load_file(training_file)
    for i in range(1000000, 100000000,100000):
        frequency_threshold = i
        training_pred_labels = [0] * len(training_words)
        for index in range(len(training_words)):
            training_frequency = counts.setdefault(training_words[index], 0)
            if training_frequency >= frequency_threshold:
                training_pred_labels[index] = 0
            else:
                training_pred_labels[index] = 1
        tprecision = get_precision(training_pred_labels, training_labels)
        trecall = get_recall(training_pred_labels, training_labels)
        tfscore = get_fscore(training_pred_labels, training_labels)
        training_performance = [tprecision, trecall, tfscore]
        if tfscore > training_max:
            training_max = tfscore
            training_threshold = i
            final_training_performance = training_performance

        dev_words, dev_labels = load_file(development_file)
        dev_pred_labels = [0] * len(dev_words)
        for index in range(len(dev_words)):
            dev_frequency = counts.setdefault(dev_words[index], 0)
            if dev_frequency >= frequency_threshold:
                dev_pred_labels[index] = 0
            else:
                dev_pred_labels[index] = 1
        dprecision = get_precision(dev_pred_labels, dev_labels)
        drecall = get_recall(dev_pred_labels, dev_labels)
        dfscore = get_fscore(dev_pred_labels, dev_labels)
        development_performance = [dprecision, drecall, dfscore]
        if tfscore > dev_max:
            dev_max = tfscore
            dev_threshold = i
            final_dev_performance = development_performance
    #print ("best_training_threshold:", training_threshold, final_training_performance, "best_dev_threshold:", dev_threshold, final_dev_performance)
    return final_training_performance, final_dev_performance


def naive_bayes(training_file, development_file, counts):
    ## YOUR CODE HERE
    import numpy as np
    training_words, training_labels = load_file(training_file)
    training_frequency = [0] * len(training_words)
    training_length = [0] * len(training_words)
    for index in range(len(training_words)):
        training_frequency[index] = counts.setdefault(training_words[index], 0)
        training_length[index] = len(training_words[index])
    tl = np.array(training_length)
    tf = np.array(training_frequency)
    tl_mean = np.mean(tl)
    tf_mean = np.mean(tf)
    tl_std = np.std(tl)
    tf_std = np.std(tf)
    tl_scale = [(l - tl_mean) / tl_std for l in tl]
    tf_scale = [(f - tf_mean) / tf_std for f in tf]
    X_train = np.matrix([tl_scale, tf_scale]).T
    Y = training_labels

    dev_words, dev_labels = load_file(development_file)
    dev_frequency = [0] * len(dev_words)
    dev_length = [0] * len(dev_words)
    for index in range(len(dev_words)):
        dev_frequency[index] = counts.setdefault(dev_words[index], 0)
        dev_length[index] = len(dev_words[index])
    dl = np.array(dev_length)
    df = np.array(dev_frequency)
    # dl_mean = np.mean(dl)
    # df_mean = np.mean(df)
    # dl_std = np.std(dl)
    # df_std = np.std(df)
    dl_scale = [(l - tl_mean) / tl_std for l in dl]
    df_scale = [(f - tf_mean) / tf_std for f in df]
    X_dev = np.matrix([dl_scale, df_scale]).T

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, Y)
    Y_pred = clf.predict(X_dev)

    precision = get_precision(Y_pred, dev_labels)
    recall = get_recall(Y_pred, dev_labels)
    fscore = get_fscore(Y_pred, dev_labels)
    development_performance = [precision, recall, fscore]

    return development_performance


def logistic_regression(training_file, development_file, counts):
    ## YOUR CODE HERE
    import numpy as np
    training_words, training_labels = load_file(training_file)
    training_frequency = [0] * len(training_words)
    training_length = [0] * len(training_words)
    for index in range(len(training_words)):
        training_frequency[index] = counts.setdefault(training_words[index], 0)
        training_length[index] = len(training_words[index])
    tl = np.array(training_length)
    tf = np.array(training_frequency)
    tl_mean = np.mean(tl)
    tf_mean = np.mean(tf)
    tl_std = np.std(tl)
    tf_std = np.std(tf)
    tl_scale = [(l - tl_mean) / tl_std for l in tl]
    tf_scale = [(f - tf_mean) / tf_std for f in tf]
    X_train = np.matrix([tl_scale, tf_scale]).T
    Y = training_labels

    dev_words, dev_labels = load_file(development_file)
    dev_frequency = [0] * len(dev_words)
    dev_length = [0] * len(dev_words)
    for index in range(len(dev_words)):
        dev_frequency[index] = counts.setdefault(dev_words[index], 0)
        dev_length[index] = len(dev_words[index])
    dl = np.array(dev_length)
    df = np.array(dev_frequency)
    # dl_mean = np.mean(dl)
    # df_mean = np.mean(df)
    # dl_std = np.std(dl)
    # df_std = np.std(df)
    dl_scale = [(l - tl_mean) / tl_std for l in dl]
    df_scale = [(f - tf_mean) / tf_std for f in df]
    X_dev = np.matrix([dl_scale, df_scale]).T

    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X_train, Y)
    Y_pred = clf.predict(X_dev)

    precision = get_precision(Y_pred, dev_labels)
    recall = get_recall(Y_pred, dev_labels)
    fscore = get_fscore(Y_pred, dev_labels)
    development_performance = [precision, recall, fscore]

    return development_performance


def own_model(training_file, development_file, counts):
    import syllables
    from nltk.corpus import wordnet
    import numpy as np
    training_words, training_labels = load_file(training_file)
    training_frequency = [0] * len(training_words) #feature 1
    training_length = [0] * len(training_words)  #feature 2
    training_synonyms = [0] * len(training_words)  #feature 3
    training_syllables = [0] * len(training_words)  #feature 4

    for index in range(len(training_words)):
        training_frequency[index] = counts.setdefault(training_words[index], 0)
        training_length[index] = len(training_words[index])
        temp = wordnet.synsets(training_words[index])
        for each in temp:
            training_synonyms[index] += len(each.lemma_names())
        training_syllables[index] = syllables.count_syllables(training_words[index])

    tl = np.array(training_length)
    tf = np.array(training_frequency)
    tsyn = np.array(training_synonyms)
    tsyl = np.array(training_syllables)

    tl_mean = np.mean(tl)
    tf_mean = np.mean(tf)
    tsyn_mean = np.mean(tsyn)
    tsyl_mean = np.mean(tsyl)

    tl_std = np.std(tl)
    tf_std = np.std(tf)
    tsyn_std = np.std(tsyn)
    tsyl_std = np.std(tsyl)

    tl_scale = [(l - tl_mean) / tl_std for l in tl]
    tf_scale = [(f - tf_mean) / tf_std for f in tf]
    tsyn_scale = [(syn - tsyn_mean) / tsyn_std for syn in tsyn]
    tsyl_scale = [(syl - tsyl_mean) / tsyl_std for syl in tsyl]

    X_train = np.matrix([tl_scale, tf_scale, tsyn_scale, tsyl_scale]).T
    Y = np.array(training_labels)
    #print("X_train: ", X_train)

    dev_words, dev_labels = load_file(development_file)
    dev_frequency = [0] * len(dev_words)  # feature 1
    dev_length = [0] * len(dev_words)  # feature 2
    dev_synonyms = [0] * len(dev_words)  # feature 3
    dev_syllables = [0] * len(dev_words)  # feature 4

    for index in range(len(dev_words)):
        dev_frequency[index] = counts.setdefault(dev_words[index], 0)
        dev_length[index] = len(dev_words[index])
        temp = wordnet.synsets(dev_words[index])
        for each in temp:
            dev_synonyms[index] += len(each.lemma_names())
        dev_syllables[index] = syllables.count_syllables(dev_words[index])

    dl = np.array(dev_length)
    df = np.array(dev_frequency)
    dsyn = np.array(dev_synonyms)
    dsyl = np.array(dev_syllables)

    # dl_mean = np.mean(dl)
    # df_mean = np.mean(df)
    # dsyn_mean = np.mean(dsyn)
    # dsyl_mean = np.mean(dsyl)
    #
    # dl_std = np.std(dl)
    # df_std = np.std(df)
    # dsyn_std = np.std(dsyn)
    # dsyl_std = np.std(dsyl)

    dl_scale = [(l - tl_mean) / tl_std for l in dl]
    df_scale = [(f - tf_mean) / tf_std for f in df]
    dsyn_scale = [(syn - tsyn_mean) / tsyn_std for syn in dsyn]
    dsyl_scale = [(syl - tsyl_mean) / tsyl_std for syl in dsyl]

    X_dev = np.matrix([dl_scale, df_scale, dsyn_scale, dsyl_scale]).T
    #print("X_dev: ", X_dev)
    Y_dev = np.array(dev_labels)

    from sklearn import svm
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, Y)
    Y_pred_svm = clf.predict(X_dev)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = 200, max_depth = 2, random_state = 0)
    clf.fit(X_train, Y)
    Y_pred_ran = clf.predict(X_dev)


    precision_svm = get_precision(Y_pred_svm, dev_labels)
    recall_svm = get_recall(Y_pred_svm, dev_labels)
    fscore_svm = get_fscore(Y_pred_svm, dev_labels)
    development_performance_svm = [precision_svm, recall_svm, fscore_svm]

    precision_ran = get_precision(Y_pred_ran, dev_labels)
    recall_ran = get_recall(Y_pred_ran, dev_labels)
    fscore_ran = get_fscore(Y_pred_ran, dev_labels)
    development_performance_ran = [precision_ran, recall_ran, fscore_ran]

    return development_performance_svm, development_performance_ran

def load_test_file(data_file):
    words = []
    with io.open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                #labels.append(int(line_split[1]))
            i += 1
    return words

def final_model(training_file, development_file, test_file, counts):
    import syllables
    from nltk.corpus import wordnet
    import numpy as np

    training_words, training_labels = load_file(training_file)
    training_frequency = [0] * len(training_words)  # feature 1
    training_length = [0] * len(training_words)  # feature 2
    training_synonyms = [0] * len(training_words)  # feature 3
    training_syllables = [0] * len(training_words)  # feature 4

    dev_words, dev_labels = load_file(development_file)
    dev_frequency = [0] * len(dev_words)  # feature 1
    dev_length = [0] * len(dev_words)  # feature 2
    dev_synonyms = [0] * len(dev_words)  # feature 3
    dev_syllables = [0] * len(dev_words)  # feature 4

    for index in range(len(training_words)):
        training_frequency[index] = counts.setdefault(training_words[index], 0)
        training_length[index] = len(training_words[index])
        temp = wordnet.synsets(training_words[index])
        for each in temp:
            training_synonyms[index] += len(each.lemma_names())
        training_syllables[index] = syllables.count_syllables(training_words[index])

    tl = np.array(training_length)
    tf = np.array(training_frequency)
    tsyn = np.array(training_synonyms)
    tsyl = np.array(training_syllables)

    for index in range(len(dev_words)):
        dev_frequency[index] = counts.setdefault(dev_words[index], 0)
        dev_length[index] = len(dev_words[index])
        temp = wordnet.synsets(dev_words[index])
        for each in temp:
            dev_synonyms[index] += len(each.lemma_names())
        dev_syllables[index] = syllables.count_syllables(dev_words[index])

    dl = np.array(dev_length)
    df = np.array(dev_frequency)
    dsyn = np.array(dev_synonyms)
    dsyl = np.array(dev_syllables)

    tlength = np.concatenate((tl, dl), axis=0)
    tfrequency = np.concatenate((tf, df), axis=0)
    tsynonyms = np.concatenate((tsyn, dsyn), axis=0)
    tsyllables = np.concatenate((tsyl, dsyl), axis=0)

    tlength_mean = np.mean(tlength)
    tfrequency_mean = np.mean(tfrequency)
    tsynonyms_mean = np.mean(tsynonyms)
    tsyllables_mean = np.mean(tsyllables)

    tlength_std = np.std(tlength)
    tfrequency_std = np.std(tfrequency)
    tsynonyms_std = np.std(tsynonyms)
    tsyllables_std = np.std(tsyllables)

    tlength_scale = [(l - tlength_mean) / tlength_std for l in tlength]
    tfrequency_scale = [(f - tfrequency_mean) / tfrequency_std for f in tfrequency]
    tsynonyms_scale = [(syn - tsynonyms_mean) / tsynonyms_std for syn in tsynonyms]
    tsyllables_scale = [(syl - tsyllables_mean) / tsyllables_std for syl in tsyllables]

    #print(len(tlength_scale), len(tfrequency_scale), len(tsynonyms_scale), len(tsyllables_scale))
    X_train = np.matrix([tlength_scale, tfrequency_scale, tsynonyms_scale, tsyllables_scale]).T
    Y_train = np.concatenate((training_labels, dev_labels), axis=0)
    #print(X_train)

    test_words = load_test_file(test_file)
    test_frequency = [0] * len(test_words)  # feature 1
    test_length = [0] * len(test_words)  # feature 2
    test_synonyms = [0] * len(test_words)  # feature 3
    test_syllables = [0] * len(test_words)  # feature 4

    for index in range(len(test_words)):
        test_frequency[index] = counts.setdefault(test_words[index], 0)
        test_length[index] = len(test_words[index])
        temp = wordnet.synsets(test_words[index])
        for each in temp:
            test_synonyms[index] += len(each.lemma_names())
        test_syllables[index] = syllables.count_syllables(test_words[index])

    tsl = np.array(test_length)
    tsf = np.array(test_frequency)
    tssyn = np.array(test_synonyms)
    tssyl = np.array(test_syllables)

    tsl_scale = [(l - tlength_mean) / tlength_std for l in tsl]
    tsf_scale = [(f - tfrequency_mean) / tfrequency_std for f in tsf]
    tssyn_scale = [(syn - tsynonyms_mean) / tsynonyms_std for syn in tssyn]
    tssyl_scale = [(syl - tsyllables_mean) / tsyllables_std for syl in tssyl]

    X_test = np.matrix([tsl_scale, tsf_scale, tssyn_scale, tssyl_scale]).T

    from sklearn import svm
    clf = svm.SVC(gamma='scale')

    clf.fit(X_train, Y_train)
    Y_final = clf.predict(X_test)
    np.savetxt('label.txt', Y_final.astype(np.int), fmt = "%d")

    return Y_final


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    words, labels = load_file("/Users/sophia/Desktop/sophia/BU/591W1NLP/assignment1/data/complex_words_training.txt")
    q2a_training_performance = all_complex(training_file)
    q2a_dev_performance = all_complex(development_file)
    print("q2a_training_performance: ", q2a_training_performance,"q2a_dev_performance :", q2a_dev_performance)

    q2b_training_performance, q2b_dev_performance = word_length_threshold(training_file, development_file)
    print("q2b_training_performance: ", q2b_training_performance, "q2b_dev_performance :", q2b_dev_performance)

    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
    q2c_training_performance, q2c_development_performance = word_frequency_threshold(training_file, development_file, counts)
    print("q2c_training_performance: ", q2c_training_performance, "q2c_development_performance :", q2c_development_performance)

    q3a_development_performance = naive_bayes(training_file, development_file, counts)
    print("q3a_development_performance: ", q3a_development_performance)

    q3b_development_performance = logistic_regression(training_file, development_file, counts)
    print("q3b_development_performance: ", q3b_development_performance)

    q4_development_performance_svm, q4_development_performance_ran = own_model(training_file, development_file, counts)
    print("q4_development_performance_svm: ", q4_development_performance_svm)
    print("q4_development_performance_ran: ", q4_development_performance_ran)

    Y_final = final_model(training_file, development_file, test_file, counts)
    #print("Y_final: ", Y_final)
