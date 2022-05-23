import os
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

current_folder = os.path.dirname(os.path.abspath(__file__))
abs_path = '/'.join([p for p in current_folder.split('/')[:-1]])+'/'

if __name__ == '__main__':

    # Load train and test data. 
    train_fp_features = torch.load(abs_path + 'data/Features/geo_train_features.pt')
    train_labels = torch.load(abs_path + 'data/Labels/train_labels.pt')
    test_fp_features = torch.load(abs_path + 'data/Features/geo_test_features.pt')
    test_labels = torch.load(abs_path + 'data/Labels/test_labels.pt')

    # Generate dictionaries where each key represents a feature. 
    train_dict, test_dict = {}, {}
    keys = range(len(train_fp_features[0]))
    for i in keys:
        train_dict[i] = []
        test_dict[i] = []

    for pocket_train in train_fp_features:
        for idx, feat_train in enumerate(pocket_train):
            train_dict[idx].append(float(feat_train))
                    
    for pocket_test in test_fp_features:
        for idx, feat_test in enumerate(pocket_test):
            test_dict[idx].append(float(feat_test))

    # Create dataframes based on dictionaries and standardize them. 
    train_df = pd.DataFrame(data = train_dict)
    stand_train_df = (train_df - train_df.mean()) / train_df.std()
    test_df = pd.DataFrame(data = test_dict)
    stand_test_df = (test_df - test_df.mean()) / test_df.std()

    model = LogisticRegression().fit(stand_train_df, train_labels)
    y_pred = model.predict(stand_test_df)
    y_pred_proba = model.predict_proba(stand_test_df)[:,1]

    # Metrics
    accuracy = model.score(stand_test_df, test_labels)
    auc_score = roc_auc_score(test_labels, y_pred_proba)
    print(f'Logistic regression -> Accuracy: {accuracy:.2f} | AUC: {auc_score:.3f}')