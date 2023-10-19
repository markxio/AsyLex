# we have a "sentiment analysis" i.e. a weight on is a sentence positive/negative for the claim
# ie we have an "outcome" per sentence, as a weight (btw 0 and 1)
# but we want the outcome of the case, and there may be several sentences per case
# we aggregate results to get the outcome of each case as Accepted - Rejected - Uncertain - No sentences found

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
from collections import Counter

def get_df(csv_path):
    df = pd.read_csv(csv_file_path, sep=';', index_col=None)
    number_rows = len(df)
    print(df.shape)
    duplicates = df.duplicated(subset=['decisionID'], keep=False).sum()

    print("-------------------------------------------------------------------")
    print("We have {} rows that have duplicates, i.e. only {} sentences that determine the case directly".format(duplicates,number_rows - duplicates))
    print("-------------------------------------------------------------------")

    return df

def plot_weights_distibution(df):
    # Data for plotting
    prediction_array = df['predicted_class'].values

    # plot
    myPlot = sns.displot(prediction_array, kind="kde", bw_adjust=.25)
    #x="Predicted weight", y="Sentences count",
    sns.despine()

    # save
    history_file_name = "./sentences_classifier_BERT_based/plot_weights_distribution.pdf"
    myPlot.figure.savefig(history_file_name)

def plot_class_distibution(df, file_name="", col_name=""):
    np.random.seed(19680801)

    colors = plt.cm.Accent(np.linspace(0, 1, 3))
    iter_color = iter(colors)
    df[f"{col_name}"].value_counts().plot.barh(title=f"Class distribution",
                                            ylabel="Decision outcome",
                                            color=colors,
                                            figsize=(9, 9)
                                               )

    for i, v in enumerate(df[f"{col_name}"].value_counts()):
        c = next(iter_color)
        plt.text(v, i,
                 " " + str(v) + ", " + str(round(v * 100 / df.shape[0], 2)) + "%",
                 color=c,
                 va='center',
                 fontweight='bold')
    history_file_name = f"./sentences_classifier_BERT_based/{file_name}.pdf"
    plt.savefig(history_file_name, bbox_inches='tight')

#takes in a list and outputs True if all elements in the input list
# evaluate as equal to each other using the standard equality operator and False otherwise
def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        return 2
    return top_two[0][0]

if __name__ == '__main__':
    csv_file_path = "./sentences_classifier_BERT_based/classification_outcome.csv"

    df = get_df(csv_file_path)

    prediction_array = df['predicted_class'].values

    max = prediction_array.max()
    min = prediction_array.min()
    # 0.77152634
    # 0.022790648
    print("Values of the predicted weights range from {} to {}".format(max, min))

    plot_weights_distibution(df)

    # because we mostly have extreme values, we transform the weights into a 0 for reject, 1 for accept
    # 2 is uncertain, between 0.4 and 0.6
    df1 = df.copy()
    mylist = []
    for i in prediction_array:
        if i < 0.4:
            mylist.append(int(0)) # reject
        elif i > 0.6:
            mylist.append(int(1)) # accept
        else:
            mylist.append(int(2)) # means "uncertain"

    df1['Case_predicted_outcome'] = mylist
    print(df1.head())

    df1.to_csv('./sentences_classifier_BERT_based/classification_outcome_3classes.csv', sep=';')

    # plot class distribution
    plot_class_distibution(df1, file_name="plot_class_distribution_after_inference", col_name='Case_predicted_outcome')

# if there are several sentences for the case, then we need to find a way to determine the outcome of that case
# we consider that a weight superior to 0.7 means the case is accepted
# a weight inferior to 0.4 means the case in rejected
# a weight btw 0.4-0.7 is uncertain
# we compute the mean of the weights per case
# that rule will also work if there is only one sentence per case

    df1.drop('extracted_sentences_determination', axis=1, inplace=True)
    df1.drop('predicted_class', axis=1, inplace=True)
    df1.drop('Unnamed: 0', axis=1, inplace=True)
    df1.set_index('decisionID', inplace=True)
    #print(df1.columns)
    #print(df1.head())


    grouped = df1.groupby('decisionID', group_keys=True) # each group is a df

    mydict = {}
    for decisionID, group_df in grouped:
        mylist = group_df['Case_predicted_outcome'].values.tolist()

        if len(mylist) == 1:
            case_outcome_agg = int(mylist[0])

        elif len(mylist) == 2:
            if all_equal(mylist):
                case_outcome_agg = int(mylist[0])
            else:
                case_outcome_agg = int(2)

        elif len(mylist) > 2:
            if all_equal(mylist):
                case_outcome_agg = int(mylist[0])
            else:
                case_outcome_agg = int(find_majority(mylist))

        mydict[decisionID] = case_outcome_agg # incremental save to a dict

    df2 = pd.DataFrame.from_dict(mydict, orient='index', columns=["decision_outcome"])
    print(df2.head())
    number_cases = len(df2) # 31227

    print("We have aggregated silver-standard annotations from classification for {} cases".format(number_cases))

    plot_class_distibution(df2, file_name="plot_case_outcome_distribution", col_name='decision_outcome')

    df2.to_csv('./sentences_classifier_BERT_based/silver_standard_final_outcomes.csv', sep=';')









