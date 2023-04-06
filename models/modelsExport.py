# %%
import pandas as pd, numpy as np
import os, json
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score, confusion_matrix, accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skopt import BayesSearchCV, Optimizer
from skopt.space import Real, Categorical, Integer
import xgboost as xgb

# %%
# Toggle option
SNV = False
FEATURE_SELECT = False
FEATURE_SELECTv2 = False
BALANCED = False
MY_BALANCED = False
MY_BALANCEDv2 = False
AUGMENTED = False # oversampling of neoplasia
AUGMENTEDv2 = False # actual augmented
AUGMENTEDv3 = False # augmented neoplasia only
SMOTE = False
SVMSMOTE = False
KMEANSSMOTE = False
ADASYNSMOTE = True
BORDERSMOTE = False
TEST = False

# Choose dataset
DATASET = 'raw'
FILENAME_ACC = 'metrics/raw_dataset/model_metrics_accuracy_ensemble.json'
FILENAME_RECALL = 'metrics/raw_dataset/model_metrics_recall_ensemble.json'
if TEST:
    DATASET = 'test'
    FILENAME_ACC = 'metrics/temp/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/temp/model_metrics_recall_ensemble.json'
elif BALANCED:
    DATASET = 'balanced'
    FILENAME_ACC = 'metrics/balanced_dataset/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/balanced_dataset/model_metrics_recall_ensemble.json'
    if SNV:
        DATASET = 'snv_balanced'
        FILENAME_ACC = 'metrics/balanced_dataset/model_metrics_accuracy_ensemble_snv.json'
        FILENAME_RECALL = 'metrics/balanced_dataset/model_metrics_recall_ensemble_snv.json'
        if FEATURE_SELECT:
            DATASET = 'snv_FS_balanced'
            FILENAME_ACC = 'metrics/balanced_dataset/model_metrics_accuracy_ensemble_snv_FS.json'
            FILENAME_RECALL = 'metrics/balanced_dataset/model_metrics_recall_ensemble_snv_FS.json'
elif MY_BALANCED:
    DATASET = 'my_balanced'
    FILENAME_ACC = 'metrics/my_balanced_dataset/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/my_balanced_dataset/model_metrics_recall_ensemble.json'
    if FEATURE_SELECT:
        DATASET = 'FS_my_balanced'
        FILENAME_ACC = 'metrics/my_balanced_dataset/model_metrics_accuracy_ensemble_FS.json'
        FILENAME_RECALL = 'metrics/my_balanced_dataset/model_metrics_recall_ensemble_FS.json'
        if SNV:
            DATASET = 'snv_FS_my_balanced'
            FILENAME_ACC = 'metrics/my_balanced_dataset/model_metrics_accuracy_ensemble_snv_FS.json'
            FILENAME_RECALL = 'metrics/my_balanced_dataset/model_metrics_recall_ensemble_snv_FS.json'
    elif FEATURE_SELECTv2:
        DATASET = 'FSv2_my_balanced'
        FILENAME_ACC = 'metrics/my_balanced_dataset/model_metrics_accuracy_ensemble_FSv2.json'
        FILENAME_RECALL = 'metrics/my_balanced_dataset/model_metrics_recall_ensemble_FSv2.json'
    elif SNV:
        DATASET = 'snv_my_balanced'
        FILENAME_ACC = 'metrics/my_balanced_dataset/model_metrics_accuracy_ensemble_snv.json'
        FILENAME_RECALL = 'metrics/my_balanced_dataset/model_metrics_recall_ensemble_snv.json'
elif AUGMENTED:
    DATASET = 'augmented'
    FILENAME_ACC = 'metrics/augmented_dataset/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/augmented_dataset/model_metrics_recall_ensemble.json'
    if FEATURE_SELECT:
        DATASET = 'augmented_FS'
        FILENAME_ACC = 'metrics/augmented_dataset/model_metrics_accuracy_ensemble_FS.json'
        FILENAME_RECALL = 'metrics/augmented_dataset/model_metrics_recall_ensemble_FS.json'
elif AUGMENTEDv2:
    DATASET = 'augmentedv2'
    FILENAME_ACC = 'metrics/augmentedv2_dataset/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/augmentedv2_dataset/model_metrics_recall_ensemble.json'
    if FEATURE_SELECT:
        DATASET = 'augmentedv2_FS'
        FILENAME_ACC = 'metrics/augmentedv2_dataset/model_metrics_accuracy_ensemble_FS.json'
        FILENAME_RECALL = 'metrics/augmentedv2_dataset/model_metrics_recall_ensemble_FS.json'
elif AUGMENTEDv3:
    DATASET = 'augmentedv3'
    FILENAME_ACC = 'metrics/augmentedv3_dataset/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/augmentedv3_dataset/model_metrics_recall_ensemble.json'
    if SNV:
        DATASET = 'snv_augmentedv3'
        FILENAME_ACC = 'metrics/augmentedv3_dataset/model_metrics_accuracy_ensemble_snv.json'
        FILENAME_RECALL = 'metrics/augmentedv3_dataset/model_metrics_recall_ensemble_snv.json'
    if FEATURE_SELECT:
        DATASET = 'augmentedv3_FS'
        FILENAME_ACC = 'metrics/augmentedv3_dataset/model_metrics_accuracy_ensemble_FS.json'
        FILENAME_RECALL = 'metrics/augmentedv3_dataset/model_metrics_recall_ensemble_FS.json'
elif SMOTE:
    DATASET = 'smote'
    FILENAME_ACC = 'metrics/smote/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/smote/model_metrics_recall_ensemble.json'
elif SVMSMOTE:
    DATASET = 'svmsmote'
    FILENAME_ACC = 'metrics/svmsmote/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/svmsmote/model_metrics_recall_ensemble.json'
    if FEATURE_SELECT:
        DATASET = 'FS_svmsmote'
        FILENAME_ACC = 'metrics/svmsmote/model_metrics_accuracy_ensemble_FS.json'
        FILENAME_RECALL = 'metrics/svmsmote/model_metrics_recall_ensemble_FS.json'
        if SNV:
            DATASET = 'snv_FS_svmsmote'
            FILENAME_ACC = 'metrics/svmsmote/model_metrics_accuracy_ensemble_snv_FS.json'
            FILENAME_RECALL = 'metrics/svmsmote/model_metrics_recall_ensemble_snv_FS.json'
    if SNV:
            DATASET = 'snv_svmsmote'
            FILENAME_ACC = 'metrics/svmsmote/model_metrics_accuracy_ensemble_FS.json'
            FILENAME_RECALL = 'metrics/svmsmote/model_metrics_recall_ensemble_FS.json'
elif KMEANSSMOTE:
    DATASET = 'kmeanssmote'
    FILENAME_ACC = 'metrics/kmeanssmote/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/kmeanssmote/model_metrics_recall_ensemble.json'
elif ADASYNSMOTE:
    DATASET = 'adasynsmote'
    FILENAME_ACC = 'metrics/adasynsmote/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/adasynsmote/model_metrics_recall_ensemble.json'
elif BORDERSMOTE:
    DATASET = 'bordersmote'
    FILENAME_ACC = 'metrics/bordersmote/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/bordersmote/model_metrics_recall_ensemble.json'
elif MY_BALANCEDv2:
    DATASET = 'my_balancedv2'
    FILENAME_ACC = 'metrics/my_balancedv2_dataset/model_metrics_accuracy_ensemble.json'
    FILENAME_RECALL = 'metrics/my_balancedv2_dataset/model_metrics_recall_ensemble.json'
elif SNV:
    DATASET = 'snv_raw'
    FILENAME_ACC = 'metrics/raw_dataset/model_metrics_accuracy_snv.json'
    FILENAME_RECALL = 'metrics/raw_dataset/model_metrics_recall_snv.json'
elif FEATURE_SELECT:
    DATASET = 'feature_select'
    FILENAME_ACC = 'metrics/selected_features/model_metrics_accuracy.json'
    FILENAME_RECALL = 'metrics/selected_features/model_metrics_recall.json'
elif FEATURE_SELECTv2:
    DATASET = 'feature_selectv2'
    FILENAME_ACC = 'metrics/selected_features/model_metrics_accuracy2.json'
    FILENAME_RECALL = 'metrics/selected_features/model_metrics_recall2.json'

DATASET

# %%
if MY_BALANCED:
    print('my_balanced')
    # x_train
    training_data = pd.read_csv('../data/balanced_data/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/balanced_data/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/balanced_data/test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/balanced_data/test_label.csv', header = None)
elif BALANCED:
    print('balanced')
    # x_train
    training_data = pd.read_csv('../data/original_data/balanced_train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/original_data/balanced_train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/balanced_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/balanced_test_label.csv', header = None)
elif AUGMENTED:
    print('augmented')
    # x_train
    training_data = pd.read_csv('../data/augmented_data/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/augmented_data/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)
elif AUGMENTEDv2:
    print('augmentedv2')
    # x_train
    training_data = pd.read_csv('../data/augmented_datav2/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/augmented_datav2/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)
elif AUGMENTEDv3:
    print('augmentedv3')
    # x_train
    training_data = pd.read_csv('../data/augmented_datav3/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/augmented_datav3/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)
elif MY_BALANCEDv2:
    print('my_balancedv2')
    # x_train
    training_data = pd.read_csv('../data/balancedv2_data/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/balancedv2_data/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)
elif SMOTE:
    print('smote')
    # x_train
    training_data = pd.read_csv('../data/SMOTE/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/SMOTE/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)
elif SVMSMOTE:
    print('svmsmote')
    # x_train
    training_data = pd.read_csv('../data/SMOTE/svm/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/SMOTE/svm/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)
elif KMEANSSMOTE:
    print('kmeanssmote')
    # x_train
    training_data = pd.read_csv('../data/SMOTE/kmeans/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/SMOTE/kmeans/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)
elif ADASYNSMOTE:
    print('adasynsmote')
    # x_train
    training_data = pd.read_csv('../data/SMOTE/adasyn/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/SMOTE/adasyn/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)
elif BORDERSMOTE:
    print('bordersmote')
    # x_train
    training_data = pd.read_csv('../data/SMOTE/border/train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/SMOTE/border/train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)
else:
    print('raw')
    # x_train
    training_data = pd.read_csv('../data/original_data/noExclusion_train_data.csv', header = None)
    # y_train
    training_labels = pd.read_csv('../data/original_data/noExclusion_train_label.csv', header = None)
    # x_test
    testing_data = pd.read_csv('../data/original_data/noExclusion_test_data.csv', header = None)
    # y_test
    testing_labels = pd.read_csv('../data/original_data/noExclusion_test_label.csv', header = None)



# %%
print(f"training_data: {type(training_data)}, \ntraining_labels: {type(training_labels)}, \ntesting_data: {type(testing_data)}, \ntesting_labels: {type(testing_labels)}")
print(f"training labels vc: \n{training_labels.value_counts()}, \ntesting labels vc: \n{testing_labels.value_counts()}")
print(len(training_data), len(training_labels), len(testing_data), len(testing_labels))

# %%
# type cast labels to ints
training_labels[0] = training_labels[0].astype(int)
# testing_labels
testing_labels[0] = testing_labels[0].astype(int)

# encode labels, using sklearn, to pass to xgboost
# this code was inspired by the snippet from:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
le = LabelEncoder()
# fit the classes to the encoder and transform labels to be 0,1,2
training_labels = le.fit_transform(training_labels[0].to_list())
testing_labels = le.fit_transform(testing_labels[0].to_list())

# gridsearch results
gs_results = {}
np.unique(training_labels)

# %%
# RUN THIS TO APPLY FEATURE SELECTION TO TRAINING DATA
if FEATURE_SELECT == True:
    ADD_POSSIBLE_FIGURES = True

    # figures contain features from (figure_num*4)+1 
    selected_figures = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                        21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,
                        39,40,41,42,43,44,45,46,47,48,49,50,51,52]

    # figures that MAY be decent - noisy but different peaks
    possible_figures = [53,54,56,57,58,59,60,61,62,63,64,65]

    # decent features - eyeballed

    # generate set of selected features
    selected_features = []

    for figure_num in selected_figures:
        for i in range(0,4):
            # print(f"figure: {figure_num}, feature: feature_{(figure_num*4)+i}")
            selected_features.append((figure_num*4)+i)

    # to add possible features
    if ADD_POSSIBLE_FIGURES == True:
        for figure_num in possible_figures:
            for i in range(0,4):
                # print(f"figure: {figure_num}, feature: feature_{(figure_num*4)+i}")
                selected_features.append((figure_num*4)+i)
                
    training_data = training_data[selected_features]
    testing_data = testing_data[selected_features]
else:
    print('no feature selection')

# %%
if FEATURE_SELECTv2:    
    # selected features round 2
    selected_features = []
    a = np.arange(30,85).tolist()
    b = np.arange(203,235).tolist()

    selected_features = np.concatenate([a,b]).tolist()
    print(selected_features)
    # (Cena, 2018)
    training_data = training_data[selected_features]
    testing_data = testing_data[selected_features]
else:
    print('no FSv2')

# %%
type(training_data)
training_data.shape

# %%
# apply SNV to training data - inspired by code from my ML CW
# (Hamzah Hafejee, 2022, COMP3611_Coursework_Assessment.ipynb, Comp 3611, University of Leeds)
# (Sklearn, 2023)
if SNV == True:
    print(len(training_data))
    # fit to training data
    scaler = StandardScaler().fit(training_data)
    training_data = scaler.transform(training_data)
    testing_data = scaler.transform(testing_data)
    print("After: \n", len(training_data))
    len(training_data)
else:
    print('no SNV standardisation')


# %%
# TO USE AVG RECALL AS METRIC FOR GS
# (gunes, 2019)
gs_recall = make_scorer(recall_score, average='macro')
DATASET

# %%
# make CART classifier
clf_cart = tree.DecisionTreeClassifier(criterion="gini", random_state=1)
# find optimal parameter values for CART
params = {
    'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40], # control overfitting,
    'max_features': [None, 'sqrt', 'log2'] # performance 
}

if DATASET == 'raw':
    # raw dataset
    params = {
        'max_depth': [None], # control overfitting,
        'max_features': ['log2'] # performance 
    }
elif DATASET == 'feature_select':
    # selected + possible features
    params = {
        'max_depth': [5], # control overfitting,
        'max_features': [None] # performance 
    }
elif DATASET == 'feature_selectv2':
    # feature select v2
    params = {'max_depth': [None], 'max_features': ['sqrt']} 
elif DATASET == 'snv_raw':
    # SNV + raw
    params = {'max_depth': [None], 'max_features': ['log2']} 
elif DATASET == 'balanced':
    params = {'max_depth': [10], 'max_features': ['sqrt']} 
elif DATASET == 'snv_balanced':
    params = {'max_depth': [10], 'max_features': ['sqrt']} 

elif DATASET == 'my_balanced':
    params = {'max_depth': [None], 'max_features': ['sqrt']} 

elif DATASET == 'FS_my_balanced':
    params = {'max_depth': [None], 'max_features': ['sqrt']} 
    
elif DATASET == 'FSv2_my_balanced':
    params = {'max_depth': [10], 'max_features': ['log2']} 
    
elif DATASET == 'snv_my_balanced':
    params = {'max_depth': [None], 'max_features': ['sqrt']} 

elif DATASET == 'snv_FS_my_balanced':
    params = {'max_depth': [5], 'max_features': [None]} 
   
elif DATASET == 'augmented':
    params = {'max_depth': [10], 'max_features': ['log2']} 

elif DATASET == 'augmented_FS':
    params = {'max_depth': [10], 'max_features': ['sqrt']} 

elif DATASET == 'smote':
    params = {'max_depth': [15], 'max_features': ['sqrt']} 

elif DATASET == 'augmentedv3':
    params = {'max_depth': [None], 'max_features': ['log2']}
    
elif DATASET == 'snv_augmentedv3':
    params = {'max_depth': [None], 'max_features': ['log2']}
 
elif DATASET == 'snv_FS_my_balanced':
    params = {'max_depth': [10], 'max_features': [None]} 

elif DATASET == 'augmentedv3_FS':
    params = {'max_depth': [None], 'max_features': ['log2']}

elif DATASET == 'kmeanssmote':
    params = {'max_depth': [10], 'max_features': ['sqrt']} 
    
elif DATASET == 'svmsmote':
    params = {'max_depth': [None], 'max_features': ['log2']}
    
elif DATASET == 'adasynsmote':
    params = {'max_depth': [None], 'max_features': ['sqrt']} 
     
elif DATASET == 'bordersmote':
    params = {'max_depth': [None], 'max_features': ['sqrt']} 
     
elif DATASET == 'snv_svmsmote':
    params = {'max_depth': [None], 'max_features': ['log2']}
    
elif DATASET == 'snv_FS_svmsmote':
    params = {'max_depth': [10], 'max_features': [None]} 
    
grid_search = GridSearchCV(clf_cart, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, np.ravel(training_labels))
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
best_cart = grid_search.best_estimator_
gs_results['CART'] = {'accuracy':best_score, 'params':best_params}

# %%
# # bayes search experiment (Skopt, 2017)
# clf_cart = tree.DecisionTreeClassifier(criterion="gini", random_state=1)
# # find optimal parameter values for CART
# params = {
#     'max_depth': Integer(1,100), # control overfitting,
#     'max_features': Categorical([None, 'sqrt', 'log2']) # performance 
# }
   
# bayes_search = BayesSearchCV(clf_cart, params, scoring='accuracy', cv=10, random_state=1)
# _ = bayes_search.fit(training_data, np.ravel(training_labels))
# best_params = bayes_search.best_params_
# best_score = bayes_search.best_score_
# print(f"params = {best_params} \nbest_score: {best_score}")
# best_cart = bayes_search.best_estimator_
# gs_results['CART'] = {'accuracy':best_score, 'params':best_params}

# %%
# make Gaussian Naive Bayes classifier
clf_nb = GaussianNB()
params = {
    'var_smoothing':[1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9], # from less smoothing to more aggressive smoothing
}
grid_search = GridSearchCV(clf_nb, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, np.ravel(training_labels))
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
best_nb = grid_search.best_estimator_
gs_results['GNB'] = {'accuracy':best_score, 'params':best_params}

# %%
# make k-Nearest Neighbours classifier
clf_knn = KNeighborsClassifier(n_jobs=-1) # use all processes for parellelisation
params = {
    'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
}
grid_search = GridSearchCV(clf_knn, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, np.ravel(training_labels))
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
best_knn = grid_search.best_estimator_
gs_results['kNN'] = {'accuracy':best_score, 'params':best_params}

# %%
# make SVM-RBF classifier
clf_svmrbf = SVC(kernel='rbf', random_state=1)
params = {
    'C': [0.1, 1, 10, 100, 1000, 5000], # high to low regularisation strength
    'gamma' : ['scale', 'auto'], # need to research this parameter more
}

if DATASET == 'raw':
    # raw dataset
    params = {
        'C': [100], # high to low regularisation strength
        'gamma' : ['scale'], # need to research this parameter more
    }

elif DATASET == 'feature_select':
    # selected + possible features
    params = {
        'C': [1000], # high to low regularisation strength
        'gamma' : ['scale'], # need to research this parameter more
    }

elif DATASET == 'feature_selectv2':
    # selected + possible features v2
    params = {
        'C': [1000], # high to low regularisation strength
        'gamma' : ['scale'], # need to research this parameter more
    }

elif DATASET == 'snv_raw':
    # SNV + raw
    params = {'C': [10], 'gamma': ['scale']} 

elif DATASET == 'balanced':
    params = {'C': [1000], 'gamma': ['scale']} 
    
elif DATASET == 'snv_balanced':
    params = {'C': [1000], 'gamma': ['scale']}

elif DATASET == 'my_balanced':
    params = {'C': [100], 'gamma': ['scale']} 

elif DATASET == 'FS_my_balanced':
    params = {'C': [100], 'gamma':[ 'scale']} 

elif DATASET == 'FSv2_my_balanced':
    params = {'C': [5000], 'gamma':[ 'scale']} 
    
elif DATASET == 'snv_my_balanced':
    params = {'C': [100], 'gamma': ['auto']} 

elif DATASET == 'snv_FS_my_balanced':
    params = {'C': [5000], 'gamma':[ 'scale']} 
    
elif DATASET == 'augmented':
    params = {'C': [5000], 'gamma': ['auto']} 

elif DATASET == 'augmented_FS':
    params = {'C': [5000], 'gamma': ['scale']} 

elif DATASET == 'smote':
    params = {'C': [5000], 'gamma': ['scale']} 

elif DATASET == 'augmentedv3':
    params = {'C': [5000], 'gamma': ['scale']} 
    
elif DATASET == 'snv_augmentedv3':
    params = {'C': [5000], 'gamma': ['scale']} 
    
elif DATASET == 'snv_FS_my_balanced':
    params = {'C': [5000], 'gamma': ['scale']} 

elif DATASET == 'augmentedv3_FS':
    params = {'C': [5000], 'gamma': ['scale']} 
    
elif DATASET == 'kmeanssmote':
    params = {'C': [5000], 'gamma': ['scale']} 
    
elif DATASET == 'svmsmote':
    params = {'C': [5000], 'gamma': ['scale']} 
    
elif DATASET == 'adasynsmote':
    params = {'C': [5000], 'gamma': ['scale']} 
    
elif DATASET == 'bordersmote':
    params = {'C': [5000], 'gamma': ['scale']} 
    
elif DATASET == 'snv_svmsmote':
    params = {'C': [1000], 'gamma': ['scale']} 
    
elif DATASET == 'snv_FS_svmsmote':
    params = {'C': [5000], 'gamma': ['auto']} 
    
grid_search = GridSearchCV(clf_svmrbf, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, np.ravel(training_labels))
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
best_svmrbf = grid_search.best_estimator_
gs_results['SVM-RBF'] = {'accuracy':best_score, 'params':best_params}

# %%
# make SVM linear classifier
clf_lin = SVC(kernel='linear', random_state=1)
params = {
    'C': [0.05, 0.1, 1, 10, 100, 1000], # high to low regularisation strength
    'gamma' : ['scale', 'auto'], # need to research this parameter more
    # 'gamma' : [], # need to research this parameter more
}

if DATASET == 'raw':
    # raw dataset
    params = {
        'C': [10], # high to low regularisation strength
        'gamma' : ['scale'], # need to research this parameter more
    }

elif DATASET == 'feature_select':
    # selected + possible features
    params = {
        'C': [1], # high to low regularisation strength
        'gamma' : ['scale'], # need to research this parameter more
    }

elif DATASET == 'feature_selectv2':
    # FSv2
    params = {
        'C': [1], # high to low regularisation strength
        'gamma' : ['scale'], # need to research this parameter more
    }

elif DATASET == 'snv_raw':
    # SNV + raw
    params = {'C': [0.1], 'gamma': ['scale']} 

elif DATASET == 'balanced':
    params = {'C': [1], 'gamma': ['scale']} 

elif DATASET == 'snv_balanced':
    params = {'C': [0.1], 'gamma': ['scale']} 

elif DATASET == 'my_balanced':
    params = {'C': [10], 'gamma': ['scale']} 

elif DATASET == 'FS_my_balanced':
    params = {'C': [10], 'gamma': ['scale']} 
    
elif DATASET == 'FSv2_my_balanced':
    params = {'C': [1000], 'gamma':[ 'scale']} 
    
elif DATASET == 'snv_my_balanced':
    params = {'C': [0.05], 'gamma': ['scale']} 

elif DATASET == 'snv_FS_my_balanced':
    params = {'C': [0.1], 'gamma': ['scale']} 
    
elif DATASET == 'augmented':
    params = {'C': [10], 'gamma': ['scale']} 

elif DATASET == 'augmented_FS':
    params = {'C': [100], 'gamma': ['scale']} 
    
elif DATASET == 'smote':
    params = {'C': [1000], 'gamma':[ 'scale']} 
    
elif DATASET == 'augmentedv3':
    params = {'C': [1000], 'gamma':[ 'scale']} 

elif DATASET == 'snv_augmentedv3':
    params = {'C': [100], 'gamma': ['scale']} 

elif DATASET == 'snv_FS_my_balanced':
    params = {'C': [1], 'gamma': ['scale']} 

elif DATASET == 'augmentedv3_FS':
    params = {'C': [1000], 'gamma':[ 'scale']} 
    
elif DATASET == 'svmsmote':
    params = {'C': [1000], 'gamma':[ 'scale']} 
        
elif DATASET == 'kmeanssmote':
    params = {'C': [10], 'gamma': ['scale']} 

elif DATASET == 'adasynsmote':
    params = {'C': [1000], 'gamma':[ 'scale']} 
    
elif DATASET == 'bordersmote':
    params = {'C': [1000], 'gamma':[ 'scale']} 
    
elif DATASET == 'snv_svmsmote':
    params = {'C': [100], 'gamma':[ 'scale']} 
    
elif DATASET == 'snv_FS_svmsmote':
    params = {'C': [100], 'gamma':[ 'scale']} 
    
grid_search = GridSearchCV(clf_lin, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, np.ravel(training_labels))
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
best_svmlin = grid_search.best_estimator_
gs_results['SVM-Lin'] = {'accuracy':best_score, 'params':best_params}

# %%
# make svm sigmoidal classifier
clf_sig = SVC(kernel='sigmoid', random_state=1)
params = {
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100], # high to low regularisation strength
    'gamma' : ['scale', 'auto'], # need to research this parameter more
    # 'gamma' : [], # need to research this parameter more
}

if DATASET == 'raw':
    # raw dataset
    params = {
        'C': [10], # high to low regularisation strength
        'gamma' : ['auto'], # need to research this parameter more
    }

elif DATASET == 'feature_select':
    # selected + possible features
    params = {
        'C': [1e-05], # high to low regularisation strength
        'gamma' : ['scale'], # need to research this parameter more
    }

elif DATASET == 'feature_selectv2':
    # FSv2
    params = {
        'C': [0.1], # high to low regularisation strength
        'gamma' : ['auto'], # need to research this parameter more
    }

elif DATASET == 'snv_raw':
    # SNV + raw
    params = {'C': [0.1], 'gamma': ['scale']} 

elif DATASET == 'balanced':
    params = {'C': [0.1], 'gamma': ['auto']}

elif DATASET == 'snv_balanced':
    params = {'C': [0.1], 'gamma': ['scale']} 
    
elif DATASET == 'my_balanced':
    params = {'C': [10], 'gamma': ['auto']} 

elif DATASET == 'FS_my_balanced':
    params = {'C': [10], 'gamma': ['auto']} 

elif DATASET == 'FSv2_my_balanced':
    params = {'C': [1], 'gamma':[ 'auto']} 
    
elif DATASET == 'snv_my_balanced':
    params = {'C': [0.1], 'gamma': ['auto']} 

elif DATASET == 'snv_FS_my_balanced':
    params = {'C': [0.1], 'gamma': ['scale']} 
    
elif DATASET == 'augmented':
    params = {'C': [100], 'gamma': ['auto']} 

elif DATASET == 'augmented_FS':
    params = {'C': [100], 'gamma': ['auto']} 

elif DATASET == 'smote':
    params = {'C': [0.1], 'gamma': ['auto']} 
    
elif DATASET == 'augmentedv3':
    params = {'C': [100], 'gamma': ['auto']} 

elif DATASET == 'snv_augmentedv3':
    params = {'C': [0.01], 'gamma': ['scale']} 

elif DATASET == 'snv_FS_my_balanced':
    params = {'C': [1], 'gamma': ['scale']} 

elif DATASET == 'augmentedv3_FS':
    params = {'C': [0.1], 'gamma': ['auto']} 

elif DATASET == 'svmsmote':
    params = {'C': [0.1], 'gamma': ['auto']} 
    
elif DATASET == 'kmeanssmote':
    params = {'C': [100], 'gamma': ['auto']} 
    
elif DATASET == 'adasynsmote':
    params = {'C': [0.1], 'gamma': ['auto']} 
    
elif DATASET == 'bordersmote':
    params = {'C': [0.1], 'gamma': ['auto']} 
    
elif DATASET == 'snv_svmsmote':
    params = {'C': [0.1], 'gamma': ['scale']} 
    
elif DATASET == 'snv_FS_svmsmote':
    params = {'C': [0.1], 'gamma': ['scale']} 
    
grid_search = GridSearchCV(clf_sig, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, np.ravel(training_labels))
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
best_svmsig = grid_search.best_estimator_
gs_results['SVM-Sig'] = {'accuracy':best_score, 'params':best_params}

# %%
# make xgboost classifier (Piotr Płoński, 2021)
clf_xgb = xgb.XGBClassifier(random_state = 1)

params = {
    'n_estimators': [10,100, 500, 1000], # no. boosting rounds
    'max_depth': [3,5,7,10,15,20] # control overfitting
}

if DATASET == 'raw':
    # raw dataset
    params = {
        'n_estimators': [100], # no. boosting rounds
        'max_depth': [10] # control overfitting
    }

elif DATASET == 'feature_select':
    # selected + possible features
    params = {
        'n_estimators': [10], # no. boosting rounds
        'max_depth': [5] # control overfitting
    }

elif DATASET == 'feature_selectv2':
    # FSv2
    params = {
        'n_estimators': [100], # no. boosting rounds
        'max_depth': [5] # control overfitting
    }

elif DATASET == 'snv_raw':
    # SNV + raw
    params = {'max_depth': [10], 'n_estimators': [100]} 

elif DATASET == 'balanced':
    params = {'max_depth': [15], 'n_estimators': [100]} 

elif DATASET == 'snv_balanced':
    params = {'max_depth': [15], 'n_estimators': [100]} 
    
elif DATASET == 'my_balanced':
    params = {'max_depth': [15], 'n_estimators': [100]} 

elif DATASET == 'FS_my_balanced':
    params = {'max_depth': [15], 'n_estimators': [100]} 

elif DATASET == 'FSv2_my_balanced':
    params = {'max_depth': [15], 'n_estimators': [500]} 
    
elif DATASET == 'snv_my_balanced':
    params = {'max_depth': [15], 'n_estimators': [100]} 

elif DATASET == 'snv_FS_my_balanced':
    params = {'max_depth': [10], 'n_estimators': [100]} 
    
elif DATASET == 'augmented':
    params = {'max_depth': [5], 'n_estimators': [500]} 

elif DATASET == 'augmented_FS':
    params = {'max_depth': [10], 'n_estimators': [1000]} 

elif DATASET == 'smote':
    params = {'max_depth': [7], 'n_estimators': [500]} 

elif DATASET == 'augmentedv3':
    params = {'max_depth': [3], 'n_estimators': [100]} 

elif DATASET == 'snv_augmentedv3':
    params = {'max_depth': [3], 'n_estimators': [100]} 
       
elif DATASET == 'snv_FS_my_balanced':
    params = {'max_depth': [5], 'n_estimators': [500]} 

elif DATASET == 'augmentedv3_FS':
    params = {'max_depth': [7], 'n_estimators': [100]} 

elif DATASET == 'svmsmote':
    params = {'max_depth': [5], 'n_estimators': [500]} 
    
elif DATASET == 'kmeanssmote':
    params = {'max_depth': [10], 'n_estimators': [500]} 
    
elif DATASET == 'adasynsmote':
    params = {'max_depth': [5], 'n_estimators': [1000]} 
    
elif DATASET == 'bordersmote':
    params = {'max_depth': [5], 'n_estimators': [100]} 
    
elif DATASET == 'snv_svmsmote':
    params = {'max_depth': [5], 'n_estimators': [500]} 
    
elif DATASET == 'snv_FS_svmsmote':
    params = {'max_depth': [3], 'n_estimators': [1000]} 
    
grid_search = GridSearchCV(clf_xgb, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, training_labels)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
best_xgb = grid_search.best_estimator_
gs_results['XGB'] = {'accuracy':best_score, 'params':best_params}

# %%
# make adaboost classifier
clf_ada = AdaBoostClassifier(random_state=1)
params = {
    'n_estimators': [10, 50, 100, 500, 1000],
    'learning_rate': [0.001, 0.01, 0.1, 1, 10] # weight applied to each clf at each boosting iteration
}

if DATASET == 'raw':
    # raw dataset
    params = {
        'n_estimators': [50],
        'learning_rate': [0.01] # weight applied to each clf at each boosting iteration
    }

elif DATASET == 'feature_select':
    # selected + possible features
    params = {
        'n_estimators': [10],
        'learning_rate': [0.01] # weight applied to each clf at each boosting iteration
    }

elif DATASET == 'feature_selectv2':
    # FSv2
    params = {
        'n_estimators': [50],
        'learning_rate': [0.01] # weight applied to each clf at each boosting iteration
    }

elif DATASET == 'snv_raw':
    # SNV + raw
    params = {'learning_rate': [0.01], 'n_estimators': [50]} 

elif DATASET == 'balanced':
    params = {'learning_rate': [0.01], 'n_estimators': [10]} 

elif DATASET == 'snv_balanced':
    params = {'learning_rate': [0.01], 'n_estimators': [10]} 
    
elif DATASET == 'my_balanced':
    params = {'learning_rate': [0.01], 'n_estimators': [100]} 

elif DATASET == 'FS_my_balanced':
    params = {'learning_rate': [0.01], 'n_estimators': [100]} 

elif DATASET == 'FSv2_my_balanced':
    params = {'learning_rate': [0.1], 'n_estimators': [10]} 
    
elif DATASET == 'snv_my_balanced':
    params = {'learning_rate': [0.001], 'n_estimators': [1000]} 

elif DATASET == 'snv_FS_my_balanced':
    params = {'learning_rate': [0.001], 'n_estimators': [1000]} 
    
elif DATASET == 'augmented':
    params = {'learning_rate': [0.01], 'n_estimators': [50]} 

elif DATASET == 'augmented_FS':
    params = {'learning_rate': [0.01], 'n_estimators': [50]} 

elif DATASET == 'smote':
    params = {'learning_rate': [0.001], 'n_estimators': [1000]} 
    
elif DATASET == 'augmentedv3':
    params = {'learning_rate': [0.001], 'n_estimators': [1000]} 

elif DATASET == 'snv_augmentedv3':
    params = {'learning_rate': [0.001], 'n_estimators': [1000]} 
        
elif DATASET == 'snv_FS_my_balanced':
    params = {'learning_rate': [0.001], 'n_estimators': [1000]} 

elif DATASET == 'augmentedv3_FS':
    params = {'learning_rate': [0.001], 'n_estimators': [1000]} 
    
elif DATASET == 'svmsmote':
    params = {'learning_rate': [0.01], 'n_estimators': [1000]} 
    
elif DATASET == 'kmeanssmote':
    params = {'learning_rate': [0.1], 'n_estimators': [10]} 
    
elif DATASET == 'adasynsmote':
    params = {'learning_rate': [1], 'n_estimators': [500]} 
    
elif DATASET == 'bordersmote':
    params = {'learning_rate': [1], 'n_estimators': [500]} 
    
elif DATASET == 'snv_svmsmote':
    params = {'learning_rate': [0.01], 'n_estimators': [1000]} 
    
elif DATASET == 'snv_FS_svmsmote':
    params = {'learning_rate': [0.1], 'n_estimators': [10]} 
    
grid_search = GridSearchCV(clf_ada, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, training_labels)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
best_ada = grid_search.best_estimator_
gs_results['ADA'] = {'accuracy':best_score, 'params':best_params}

# %%
# make Logistic Regressor
clf_lr = LogisticRegression(random_state=1, max_iter=1000)
params = {
    'penalty': ['l1', 'l2'], # type of regularisation 
    'C': [0.1, 1, 10, 100], # regularisation strength
    'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'] # approach to finding best weights
}

print(DATASET)

if DATASET == 'raw':
    # raw dataset
    params = {
        'penalty': ['l2'], # type of regularisation 
        'C': [0.1], # regularisation strength
        'solver': ['lbfgs'] # approach to finding best weights
    }
elif DATASET == 'feature_select':
    # selected + possible features
    params = {
        'penalty': ['l2'], # type of regularisation 
        'C': [0.1], # regularisation strength
        'solver': ['newton-cg'] # approach to finding best weights
    }
elif DATASET == 'feature_selectv2':
    # FSv2
    params = {'C': [1], 'penalty': ['l2'], 'solver': ['lbfgs']} 
elif DATASET == 'snv_raw':
    # SNV + raw
    params = {'C': [0.1], 'penalty': ['l2'], 'solver': ['lbfgs']} 
elif DATASET == 'balanced':
    params = {'C': [100], 'penalty': ['l2'], 'solver': ['saga']} 
elif DATASET == 'snv_balanced':
    params = {'C': [0.1], 'penalty': ['l2'], 'solver': ['saga']} 

elif DATASET == 'my_balanced':
    params = {'C': [10], 'penalty': ['l2'], 'solver': ['saga']} 
    
elif DATASET == 'FS_my_balanced':
    params = {'C': [10], 'penalty': ['l2'], 'solver': ['saga']} 
    
elif DATASET == 'FSv2_my_balanced':
    params = {'C': [10], 'penalty': ['l2'], 'solver': ['saga']} 
    
elif DATASET == 'snv_my_balanced':
    params = {'C': [1], 'penalty': ['l1'], 'solver': ['saga']} 
    
elif DATASET == 'snv_FS_my_balanced':
    params = {'C': [1], 'penalty': ['l2'], 'solver': ['saga']} 
    
elif DATASET == 'augmented':
    params = {'C': [10], 'penalty': ['l2'], 'solver': ['lbfgs']} 
    
elif DATASET == 'augmented_FS':
    params = {'C': [100], 'penalty': ['l1'], 'solver': ['liblinear']} 

elif DATASET == 'smote':
    params = {'C': [10], 'penalty': ['l2'], 'solver': ['saga']} 

elif DATASET == 'augmentedv3':
    params = {'C': [100], 'penalty': ['l1'], 'solver': ['liblinear']} 
    
elif DATASET == 'snv_augmentedv3':
    params = {'C': [10], 'penalty': ['l2'], 'solver': ['lbfgs']} 
    
elif DATASET == 'snv_FS_my_balanced':
    params = {'C': [10], 'penalty': ['l1'], 'solver': ['saga']} 

elif DATASET == 'augmentedv3_FS':
    params = {'C': [100], 'penalty': ['l1'], 'solver': ['liblinear']} 
    
elif DATASET == 'svmsmote':
    params = {'C': [100], 'penalty': ['l2'], 'solver': ['lbfgs']} 
    
elif DATASET == 'kmeanssmote':
    params = {'C': [10], 'penalty': ['l2'], 'solver': ['lbfgs']} 
    
elif DATASET == 'adasynsmote':
    params = {'C': [100], 'penalty': ['l1'], 'solver': ['liblinear']} 
    
elif DATASET == 'bordersmote':
    params = {'C': [100], 'penalty': ['l1'], 'solver': ['liblinear']} 
    
elif DATASET == 'snv_svmsmote':
    params = {'C': [100], 'penalty': ['l2'], 'solver': ['lbfgs']} 
    
elif DATASET == 'snv_FS_svmsmote':
    params = {'C': [100], 'penalty': ['l2'], 'solver': ['lbfgs']} 
    
grid_search = GridSearchCV(clf_lr, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, np.ravel(training_labels))
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
gs_results['LR'] = {'accuracy':best_score, 'params':best_params}


# %%
print(f"params = {best_params} \nbest_score: {best_score}")
best_lr = grid_search.best_estimator_

# %%
# make Random Forest classifier
clf_rf = RandomForestClassifier(random_state=1)
params = {
  'n_estimators': [10, 50, 100, 200, 300],
  'max_depth': [None, 5, 10, 20, 30, 40],
  "max_features" : [None, 1, 5, 10, 20, 30]
}

if DATASET == 'raw':
  # raw dataset
  params = {
    'n_estimators': [50],
    'max_depth': [10],
    "max_features" : [30]
  }

elif DATASET == 'feature_select':
  # selected + possible features
  params = {
    'n_estimators': [200],
    'max_depth': [None],
    "max_features" : [5]
  }

elif DATASET == 'feature_selectv2':
  # feature select v2
  params = {'max_depth': [None], 'max_features': [5], 'n_estimators': [300]}

elif DATASET == 'snv_raw':
  # SNV + raw
  params = {'max_depth': [10], 'max_features': [30], 'n_estimators': [50]} 

elif DATASET == 'balanced':
  params = {'max_depth': [10], 'max_features': [5], 'n_estimators': [100]} 

elif DATASET == 'snv_balanced':
  params = {'max_depth': [10], 'max_features': [5], 'n_estimators': [100]} 
  
elif DATASET == 'my_balanced':
  params = {'max_depth': [10], 'max_features': [5], 'n_estimators': [300]} 

elif DATASET == 'FS_my_balanced':
  params = {'max_depth': [10], 'max_features': [5], 'n_estimators': [300]} 

elif DATASET == 'FSv2_my_balanced':
  params = {'max_depth': [None], 'max_features': [30], 'n_estimators': [100]} 
    
elif DATASET == 'snv_my_balanced':
  params = {'max_depth': [10], 'max_features': [5], 'n_estimators': [300]}

elif DATASET == 'snv_FS_my_balanced':
  params = {'max_depth': [None], 'max_features': [10], 'n_estimators': [10]} 

elif DATASET == 'augmented':
  params = {'max_depth': [None], 'max_features': [20], 'n_estimators': [100]} 

elif DATASET == 'augmented_FS':
  params = {'max_depth': [None], 'max_features': [1], 'n_estimators': [50]} 

elif DATASET == 'smote':
  params = {'max_depth': [None], 'max_features': [5], 'n_estimators': [50]} 

elif DATASET == 'augmentedv3':
  params = {'max_depth': [None], 'max_features': [10], 'n_estimators': [200]} 

elif DATASET == 'snv_augmentedv3':
  params = {'max_depth': [None], 'max_features': [10], 'n_estimators': [200]} 
    
elif DATASET == 'snv_FS_my_balanced':
  params = {'max_depth': [None], 'max_features': [10], 'n_estimators': [300]} 
  
elif DATASET == 'augmentedv3_FS':
  params = {'max_depth': [10], 'max_features': [1], 'n_estimators': [300]} 

elif DATASET == 'svmsmote':
  params = {'max_depth': [None], 'max_features': [20], 'n_estimators': [50]} 
    
elif DATASET == 'kmeanssmote':
  params = {'max_depth': [None], 'max_features': [5], 'n_estimators': [100]} 
  
elif DATASET == 'adasynsmote':
  params = {'max_depth': [None], 'max_features': [5], 'n_estimators': [50]} 
    
elif DATASET == 'bordersmote':
  params = {'max_depth': [10], 'max_features': [5], 'n_estimators': [50]} 
    
elif DATASET == 'snv_svmsmote':
  params = {'max_depth': [None], 'max_features': [20], 'n_estimators': [50]} 
    
elif DATASET == 'snv_FS_svmsmote':
  params = {'max_depth': [None], 'max_features': [5], 'n_estimators': [100]} 
    
grid_search  = GridSearchCV(clf_rf, params, scoring='accuracy', cv=10)
grid_search.fit(training_data, np.ravel(training_labels))
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"params = {best_params} \nbest_score: {best_score}")
best_rf = grid_search.best_estimator_
gs_results['RF'] = {'accuracy':best_score, 'params':best_params}

# %%
# ensemble model (Sklearn, 2014), (Sklearn, 2023)
ensemble = VotingClassifier(estimators=[
    ('rf', best_rf), 
    ('knn', best_knn), 
    ('xgb', best_xgb), 
    ('svmrbf', best_svmrbf), 
    ('nb', best_nb)],
    voting='hard',
    n_jobs=-1)

ensemble.fit(training_data, training_labels)

accuracy = ensemble.score(testing_data, testing_labels)
predictions = ensemble.transform(testing_data)
gs_results['Ensemble'] = {'accuracy':accuracy}
accuracy

# %%
# sorted GS models
print(gs_results)
gs_sorted_models = dict(sorted(gs_results.items(), key=lambda item: item[1]['accuracy'], reverse=True))
print(gs_sorted_models.keys())

# %%
# evaluate models (Sklearn, 2023)
# model_metrics = {'accuracy', 'recall', 'precision', 'F1-score', 'ROC-AUC'}
model_metrics = {}

# all of the models
models = [best_cart, best_rf, best_lr, best_nb, best_knn, best_svmrbf, best_svmlin, best_svmsig, best_xgb, best_ada, ensemble]
model_names = ['CART', 'RF', 'LR', 'GNB', 'kNN', 'SVM-RBF', 'SVM-Lin', 'SVM-Sig', 'XGB', 'ADA', 'Ensemble']
i=0
for model in models:
    # train on test set
    predicted = model.predict(testing_data)
    # generate cm against test labels
    cm = confusion_matrix(testing_labels, predicted)
    # print(cm)
    accuracy = accuracy_score(testing_labels, predicted)
    recall = recall_score(testing_labels, predicted, average=None)
    precision = precision_score(testing_labels, predicted, average=None)
    f1 = f1_score(testing_labels, predicted, average=None)

    try:
        predicted_prob = model.predict_proba(testing_data)
        roc = roc_auc_score(testing_labels, predicted_prob, average=None, multi_class='ovr') 
        # print(accuracy, recall, precision, f1, roc)
        model_metrics[model_names[i]] = {
                                            'accuracy':accuracy, 
                                            'recall':{
                                                1:recall[0], 
                                                2:recall[1], 
                                                3:recall[2]
                                            },
                                            'precision':{
                                                1:precision[0], 
                                                2:precision[1], 
                                                3:precision[2]
                                            },
                                            'f1_score':{
                                                1:f1[0], 
                                                2:f1[1], 
                                                3:f1[2]
                                            },
                                            'ROC-AUC':{
                                                1:roc[0], 
                                                2:roc[1], 
                                                3:roc[2]
                                            }
        }
    except:
        print(f"can't predict class probilities for {model_names[i]}")
        # print(accuracy, recall, precision, f1)
        model_metrics[model_names[i]] = {
                                            'accuracy':accuracy, 
                                            'recall':{
                                                1:recall[0], 
                                                2:recall[1], 
                                                3:recall[2]
                                            },
                                            'precision':{
                                                1:precision[0], 
                                                2:precision[1], 
                                                3:precision[2]
                                            },
                                            'f1_score':{
                                                1:f1[0], 
                                                2:f1[1], 
                                                3:f1[2]
                                            }
        }
    i+=1

# (Gern Blanston, 2009)- sort by neoplasia recall
sorted_metrics = dict(sorted(model_metrics.items(), key=lambda item: item[1]['recall'][3], reverse=True))
# (holys, 2013)
with open(FILENAME_RECALL, 'w') as fp:
    json.dump(sorted_metrics, fp)

# redo but sort by accuracy
# (Gern Blanston, 2009)
sorted_metrics_acc = dict(sorted(model_metrics.items(), key=lambda item: item[1]['accuracy'], reverse=True))
# (holys, 2013)
with open(FILENAME_ACC, 'w') as fp:
    json.dump(sorted_metrics_acc, fp)
model_metrics
sorted_metrics
FILENAME_ACC, FILENAME_RECALL
 

# %%
# print highest acc models from gridsearch
print(f"gs_sorted_models (acc): \n{gs_sorted_models.keys()}\n")

# highest acc models from test set
print(f"sorted models (acc): \n{sorted_metrics_acc.keys()}\n")

# highest recall from test set
print(f"sorted models (recall): \n{sorted_metrics.keys()}")


# %%
DATASET

# %%
# (Neekhara, 2019)
import json
# DATASET = 'snv_svmsmote'

# function to add to JSON
def write_json(new_data, filename='metrics/scoreboard.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data[DATASET] = (new_data)
        file_data = dict(sorted(file_data.items(), key=lambda item: item[1]['all']['accuracy'], reverse=True))
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

# calculate avg accuracy and recall
accuracy = 0
recall = 0
count = 0
for model in model_names:
    # print(model_metrics[model]['accuracy'])
    accuracy += model_metrics[model]['accuracy']
    recall += model_metrics[model]['recall'][3]
    count +=1

t6acc = 0
t6rec = 0
count2 = 0
for key in sorted_metrics_acc:
    if count2 ==6:
        break
    t6acc += sorted_metrics_acc[key]['accuracy']
    t6rec += sorted_metrics_acc[key]['recall'][3]
    count2 +=1

avg = {'all' : {'accuracy': accuracy/count, 'neoplasia recall': recall/count},
       'top6' : {'accuracy': t6acc/count2, 'recall':t6rec/count2}}
print(accuracy, recall, count)
print(t6acc, t6rec, count2)
if DATASET != 'test':
    write_json(avg)
else:
    print(DATASET)
avg



# %% [markdown]
# # observations on raw dataset
# XGBoost and ADAboost seem to have really overfit, because they severely underperform on unseen test data, compared to the accuracies they were achieving with gridsearch. IGNORE THIS: it is just because the test labels were not normalised!
# 
# Although GNB has higher recall for neoplasia than kNN, kNN seems to be the best classifier overall. While GNB has highest recall for neoplasia, has 3rd lowest accuracy.
# 
# Top models based on accuracy, from gridsearch, were RF, kNN, XGB, SVM-RBF, CART. Top models based on accuracy, from test set, were RF, kNN, SVM-RBF, CART, SVM-Lin. Therefore, RF, kNN, SVM-RBF, CART seem to perform well, in terms of accuracy, and don't seem to produce drastically different results with the test set, suggesting there isn't much overfitting
# 
# gs_sorted_models (acc): 
# (['RF', 'kNN', 'XGB', 'SVM-RBF', 'CART', 'LR', 'SVM-Lin', 'ADA', 'SVM-Sig', 'GNB'])
# 
# sorted models (acc): 
# (['RF', 'XGB', 'kNN', 'SVM-RBF', 'CART', 'SVM-Lin', 'LR', 'SVM-Sig', 'GNB', 'ADA'])
# 
# sorted models (recall): 
# (['GNB', 'kNN', 'CART', 'RF', 'SVM-Lin', 'XGB', 'LR', 'SVM-RBF', 'SVM-Sig', 'ADA'])
# 
# # observations on feature selected dataset
# Some models decreased in performance, some increased, with largest increase being 6% increase in accuracy for SVM-Lin model. But overall, not worth, since the max accuracy of any of the models was lower than without feature selection. Maybe better feature selection is needed - an analytical solution rather than eyeball
# 
# gs_sorted_models (acc): 
# (['RF', 'SVM-RBF', 'kNN', 'XGB', 'CART', 'SVM-Lin', 'LR', 'ADA', 'GNB', 'SVM-Sig'])
# 
# sorted models (acc): 
# (['RF', 'XGB', 'kNN', 'SVM-RBF', 'CART', 'LR', 'SVM-Lin', 'GNB', 'ADA', 'SVM-Sig'])
# 
# sorted models (recall): 
# (['GNB', 'SVM-RBF', 'XGB', 'RF', 'kNN', 'SVM-Lin', 'CART', 'LR', 'SVM-Sig', 'ADA'])


