from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
 
def compute_cfMatrix(y_pred, y_true, labels):
    sensitivity = []
    specificity = []
    PPV = []
    NPV = []
    Accuracy = []
    F1 = []
    
    perClass_cfM = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    epsilon = 0.00001
    for i in range (len(labels)):
        TN = perClass_cfM[i][0,0]
        FN = perClass_cfM[i][1,0]
        TP = perClass_cfM[i][1,1]
        FP = perClass_cfM[i][0,1]
        # recall
        r = TP/(TP+FN+epsilon)
        sensitivity.append(r)
        # specificity
        specificity.append(TN/(TN+FP+epsilon))
        # precision
        p = TP/(TP+FP+epsilon)
        PPV.append(p)
        # F1-score
        F1.append(2*(p*r)/(p+r+epsilon))
        # NPV
        NPV.append(TN/(TN+FN+epsilon))
        
        Accuracy.append((TP+TN+epsilon)/(TP+TN+FP+FN+epsilon))

    return sensitivity, specificity, PPV, NPV, Accuracy, F1