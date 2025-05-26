
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")

import numpy as np

from typing import Literal, Union
from matplotlib import pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (roc_curve, roc_auc_score, auc, 
                            precision_recall_curve, classification_report
                            )
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


CLASSIFIER_MAP = {
    "nb": {
            "name": "Naive Bayes",
            "model": MultinomialNB,
            "hyper_param": "alpha",
        },
    "svc": {
            "name": "Linear SVM",
            "model": svm.LinearSVC,
            "hyper_param": "C"
        }
    }

# region: CLASSIFIERS

def show_plots(curve_data, model_name, lw=2):
    
    n = len(curve_data)
    # create subplots
    fig, ax = plt.subplots(1, n)
    # set fig width and title
    fig.set_figwidth(5*n)
    fig.suptitle(f"{model_name} Classifier Model Performance")
    
    for i, (curve_type, data) in enumerate(curve_data.items()):
        # plot curve points
        ax[i].plot(data["x"], data["y"], color='darkorange', lw=lw)
        ax[i].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        
        # Add a text box for the curve score
        ax[i].text(0.6, 0.1,
                s=f'{curve_type.upper()} Score: {data["score"]:.2%}', 
                bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
        
        # set axis bounds
        ax[i].set_xlim([0.0, 1.0])
        ax[i].set_ylim([0.0, 1.05])
        # set axis labels and title
        ax[i].set_xlabel(data["labels"][0])
        ax[i].set_ylabel(data["labels"][1])
        ax[i].set_title(curve_type.upper())
        
    # Adjust layout to prevent overlapping and show plot
    plt.tight_layout()
    plt.show()


def predictions(model_type, model, x_test):
    
    test_predictions = pred_probability = []
    
    if model_type == "nb":
        # get classification probs and choose "1" class
        pred_probability = model.predict_proba(x_test)[:,1]
        # convert to binary output based on confidence score
        test_predictions = np.where(pred_probability > 0.5, 1, 0)    
    
    elif model_type == "svc":
        # predict test set output
        test_predictions = model.predict(x_test)
        # get probability of guessing each class
        pred_probability = model.decision_function(x_test)
        
    return test_predictions, pred_probability


def vectorize_data(x_test, x_train, stop_words, min_df):
    # initialize and fit the TfidfVectorizer
    vec = TfidfVectorizer(stop_words=stop_words, min_df=min_df)
    vec.fit(x_test)
    # vectorize data
    return vec.transform(x_test), vec.transform(x_train)


def fit_model(x_train, y_train,
              model_type: Union[Literal["nb"], Literal["svc"]], 
              algorithm_para=1.0):
    
    # choose classification model and add hyperparams
    model_params = CLASSIFIER_MAP[model_type]
    model = model_params["model"](**{ model_params["hyper_param"]: algorithm_para })
    
    # fit classifier model with training data
    model.fit(x_train, y_train)
    
    return model, model_params["name"]


def predict_and_evaluate(model_type, model, x_test, y_test):
    
    curve_data = {
        "auc": {"labels": ("False Positive Rate", "True Positive Rate")}, 
        "prc" :{"labels": ("Recall", "Precision")}
    } 
    
    # predict test set output and get probability of guessing each class
    test_predictions, pred_probability = predictions(model_type, model, x_test)
    
    # print classification report
    print(classification_report(y_test, test_predictions, target_names=['0','1']))

    # get ROC curve using tpr/fpr and calculate AUC score
    fpr, tpr, _ = roc_curve(y_test, pred_probability)
    auc_data = { 
                    "x": fpr, "y": tpr,
                    "score": roc_auc_score(y_test, pred_probability)
                }
    curve_data["auc"].update(auc_data)
    
    # get PRC curve using precision/recall and calculate PRC score
    precision, recall, _ = precision_recall_curve(y_test, pred_probability)
    prc_data =  {
                    "x": recall, "y": precision,
                    "score": auc(recall, precision)
                }
    curve_data["prc"].update(prc_data)

    return curve_data


def classify_data(x_train, y_train, x_test, y_test, model_type, algorithm_para=1.0, min_df=1, stop_words=None):
    
    x_test, x_train = vectorize_data(x_test, x_train, stop_words, min_df)

    model, model_name = fit_model(x_train, y_train, model_type, algorithm_para)
    
    curve_data = predict_and_evaluate(model_type, model, x_test, y_test)
    show_plots(curve_data, model_name)
    
    return curve_data


def search_params(x_train, y_train, clf, vectorizer=TfidfVectorizer()):
    
    classifier = CLASSIFIER_MAP[clf]["model"]()
    hyperparam = f'clf__{CLASSIFIER_MAP[clf]["hyper_param"]}'
    
    metric = 'f1_macro'
    # initialize search params
    params = {
    'tfidf__stop_words': [None, 'english'],
    'tfidf__min_df': [1, 2, 5],
    hyperparam: [0.1, 0.5, 1]
    }
    # create pipeline with vectorizer and classifier
    pipeline = pipeline = Pipeline([
                                     ("tfidf", vectorizer),
                                     ("clf", classifier)
                                   ])

    # return optimal params
    grid = GridSearchCV(pipeline, param_grid=params, scoring=metric, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)

    min_df = grid.best_params_['tfidf__min_df']
    stop_words = grid.best_params_['tfidf__stop_words']
    C = grid.best_params_[hyperparam]

    print("Optimal Grid Search Parameters:\n"
        f"Best f1 score = {grid.best_score_:.3f}\n"
        f"Min. DF = {min_df}\n"
        f"Stopwords = {stop_words}\n"
        f"Classifier Hyper-parameter: {C}\n"
        )
    return min_df, stop_words, C


def sample_size_impact(docs, y, model_type):

    train_size = list(range(1,10))
    train_size.reverse()
    train_size = [i/10 for i in train_size]

    performance = []
    for size in train_size:
        print(f'Training sample size: {(10 - size*10) / 10}')
        # separate into train and test
        x_train, x_test, y_train, y_test = train_test_split(docs, y,  test_size=size, random_state=0)
        # calculate auc
        curve_data = classify_data(x_train, y_train, x_test, y_test, model_type)
        performance.append(curve_data["auc"]["score"])

    plt.figure().set_figwidth(5)
    
    plt.grid(True)
    plt.axis((1, 0, 0.8, 1))
    
    plt.plot(train_size, performance, color='blue', lw=2, label='Model Performance')
    
    plt.title(f"Impact of Sample Size on {CLASSIFIER_MAP[model_type]['name']} Classifier Performance")
    
    plt.xlabel('Testing Sample Percentage')
    plt.ylabel('AUC')

    plt.show()

# endregion: CLASSIFIERS