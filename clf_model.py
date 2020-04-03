import logs
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from splitData import balanceDataSet
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,classification_report
from textblob import TextBlob
from sklearn.utils import shuffle
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold


def createClassifier(clf,df_trainSet,df_testSet):
    '''
    Passes the dataset via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Trains the classifier (Random Forest / Naive Bayes / SVM / Ensemble using Voting Classifier)

    Parameters : 
    clf (str) : Name of classifier (options - RF, NB, SVM , ensemble)
    df_trainSet (DataFrame) : Training Data
    df_testSet (DataFrame) : Test Data

    Returns : 
    count_vect : Count Vectorizer Model
    tfidf_transformer : TFIDF Transformer Model
    clf_model : Trained Model 
    clf_test_score (float) : Accuracy achieved on Test Set 
    f1/precision/recall (float) : F1, Precision and Recall scores (macro average)
    '''
    
    #logs.writeLog("\nSplitting the Training/Test Set into training and test set - "+str(1-splitratio)+"/"+str(splitratio)+" split.")
    #df_trainSet,df_testSet = balanceDataSet(df_annotatedSet,"Label",1-splitratio)

    #df_trainSet = shuffle(df_trainSet)
    #df_testSet = shuffle(df_testSet)

    #Convert dataframes to numpy array's
    X_train = df_trainSet.loc[:,['req_1','req_2']]  #Using req_1,req_2 rather than req1,req2 because req_1,req_2 have been cleaned - lower case+punctuations
    y_train = df_trainSet.loc[:,'Label']
    X_test = df_testSet.loc[:,['req_1','req_2']]
    y_test = df_testSet.loc[:,'Label']

    logs.writeLog("\nTraining Set Size : "+str(len(X_train)))
    logs.writeLog("\nTrain Set Value Count : \n"+str(df_trainSet['Label'].value_counts()))

    logs.writeLog("\nTest Set Size : "+str(len(X_test)))
    logs.writeLog("\nTest Set Value Count : \n"+str(df_testSet['Label'].value_counts()))
    
    logs.writeLog("\n\nTraining Model....")
    
    #Perform Bag of Words
    count_vect = CountVectorizer(tokenizer=my_tokenizer,lowercase=False)
    X_train_counts = count_vect.fit_transform(np.array(X_train))
    
    #Transform a count matrix to a normalized tf or tf-idf representation.
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)
    
    X_test_counts = count_vect.transform(np.array(X_test))
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    
    #Initiate Classifiers
    rf_model = RandomForestClassifier()
    nb_model = MultinomialNB()
    svm_model = SVC(probability=True)  #predict_proba not available if probability = False

    #Random Forest Classifier Creation
    if clf == "RF" :
        clf_model = rf_model.fit(X_train_tfidf, np.array(y_train).astype('int'))
        
    #Naive Bayes Classifier Creation
    elif clf == "NB":
        clf_model = nb_model.fit(X_train_tfidf,np.array(y_train).astype('int'))

    #Support Vector Machine Classifier Creation.
    elif clf == "SVM":
        clf_model = svm_model.fit(X_train_tfidf,np.array(y_train).astype('int'))
    
    #Ensemble Creation
    elif clf == "ensemble":
        #Predict_proba works only when Voting = 'soft'
        #n_jobs = -1 makes allows models to be created in parallel (using all the cores, else we can mention 2 for using 2 cores)
        clf_model = VotingClassifier(estimators=[('RF', rf_model), ('NB', nb_model),('SVM',svm_model)], voting='soft',n_jobs=-1)  
        clf_model.fit(X_train_tfidf,np.array(y_train).astype('int'))

    #Predict labels
    predict_labels = clf_model.predict(X_test_tfidf)
    actualLabels = np.array(y_test).astype('int')
    labelClasses = list(set(actualLabels))   #np.array(y_train).astype('int')
    
    #Calculate Classifier Test Accuracy and other important metrics
    clf_test_score = clf_model.score(X_test_tfidf,actualLabels)
    logs.writeLog ("\n"+clf+" Classifier Test Score : "+str(clf_test_score))
    
    f1 = round(f1_score(actualLabels, predict_labels,average='macro'),2)
    precision = round(precision_score(actualLabels, predict_labels,average='macro'),2)
    recall = round(recall_score(actualLabels, predict_labels,average='macro'),2)
    
    logs.writeLog ("\n\nClassification Report on Test Set: \n\n"+str(classification_report(actualLabels,predict_labels)))
    cm = confusion_matrix(actualLabels,predict_labels,labels=labelClasses)    
    logs.writeLog ("\n\nConfusion Matrix : \n"+str(cm)+"\n")
    

    return count_vect, tfidf_transformer, clf_model,clf_test_score,f1,precision,recall  

def Crossfoldvalidation(cv,tfidf,clf_model,df_validationSet):
    '''
    Passes the validation dataset via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Performs 5 and 10-fold cross validation on the Validation Dataset and returns the scores. [mean+(+/-std_dev)]
    
    Parameters : 
    cv : Count Vectorizer Model
    tfidf : TFIDF Transformer Model
    clf_model : Trained Model 
    df_validationSet (DataFrame) : Validation Data (Unseen Data)

    Returns : 
    scores_5 (float) : 5 Fold Cross Validation Score on Validation Set 
    scores_10 (float) : 10 Fold Cross Validation Score on Validation Set
    '''
    
    #Convert Dataframe to numpy array
    predictData = np.array(df_validationSet.loc[:,['req_1','req_2']])
    actualLabels = np.array(df_validationSet.loc[:,"Label"]).astype('int')

    #Pass the data through nlp pipeline
    predict_counts = cv.transform(predictData)
    predict_tfidf = tfidf.transform(predict_counts)
    
    #Perform Cross Validation
    crossv_5 = StratifiedKFold(5)
    scores_5 = cross_val_score(clf_model, predict_tfidf, actualLabels, cv=crossv_5) #https://scikit-learn.org/stable/modules/cross_validation.html
    #logs.writeLog("Accuracy: %0.2f (+/- %0.2f)" % (scores_5.mean(), scores_5.std() * 2))
    
    crossv_10 = StratifiedKFold(10)
    scores_10 = cross_val_score(clf_model,predict_tfidf,actualLabels,cv=crossv_10)
    #logs.writeLog("Accuracy: %0.2f (+/- %0.2f)" % (scores_10.mean(), scores_10.std() * 2))
    
    scores_5 = str(round(scores_5.mean(),2)) +"(+/- "+str(round(scores_5.std()*2,2))+")"
    scores_10 = str(round(scores_10.mean(),2)) +"(+/- "+str(round(scores_10.std()*2,2))+")"
    
    return scores_5,scores_10

def predictLabels(cv,tfidf,clf,df_toBePredictedData):
    '''
    Passes the to be predicted dataset via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Predicts and returns the labels for the input data in a form of DataFrame.

    Parameters : 
    cv : Count Vectorizer Model
    tfidf : TFIDF Transformer Model
    clf : Trained Model 
    df_toBePredictedData (DataFrame) : To Be Predicted Data (Unlabelled Data)

    Returns : 
    df_toBePredictedData (DataFrame) : Updated To Be Predicted Data (Unlabelled Data), including prediction probabilities for different labels
    '''
    
    predictData = np.array(df_toBePredictedData.loc[:,['req_1','req_2']])
    #logs.writeLog(str(df_toBePredictedData))
    
    predict_counts = cv.transform(predictData)
    predict_tfidf = tfidf.transform(predict_counts)
    predict_labels = clf.predict(predict_tfidf)
    predict_prob = clf.predict_proba(predict_tfidf)
    
    logs.writeLog ("\nTotal Labels Predicted : "+ str(len(predict_labels)))

    df_toBePredictedData['predictedProb'] = predict_prob.tolist() 
    df_toBePredictedData['maxProb'] = np.amax(predict_prob,axis=1)
    
    return df_toBePredictedData    

def validateClassifier(cv,tfidf,clf_model,df_validationSet):
    '''
    Passes the validation dataset (Unseen data) via NLP Pipeline (Count Vectorizer , TFIDF Transformer)
    Calculate the accuracy and other metrics to evaluate the performance of the model on validation set (unseen data)
    
    Parameters : 
    cv : Count Vectorizer Model
    tfidf : TFIDF Transformer Model
    clf : Trained Model 
    df_validationSet (DataFrame) : Validation Data (Unseen Data)

    Returns : 
    clf_val_score/f1/precision/recall (float) : Accuracy Value on Validation Data / F1 score / Precision / Recall
    '''
    
    predictData = np.array(df_validationSet.loc[:,['req_1','req_2']])
    
    actualLabels = np.array(df_validationSet.loc[:,'Label']).astype('int')
    predict_counts = cv.transform(predictData)
    predict_tfidf = tfidf.transform(predict_counts)
    
    predict_labels = clf_model.predict(predict_tfidf)
    clf_val_score = clf_model.score(predict_tfidf,actualLabels)

    f1 = round(f1_score(actualLabels, predict_labels,average='macro'),2)
    precision = round(precision_score(actualLabels, predict_labels,average='macro'),2)
    recall = round(recall_score(actualLabels, predict_labels,average='macro'),2)
    
    labelClasses = list(set(actualLabels))   #np.array(y_train).astype('int')
    logs.writeLog ("\n\nClassification Report On Validation Set: \n\n"+str(classification_report(actualLabels,predict_labels)))
    cm = confusion_matrix(actualLabels,predict_labels,labels=labelClasses)    
    logs.writeLog ("\n\nConfusion Matrix : \n"+str(cm)+"\n")
    
    return clf_val_score,f1,precision,recall

def my_tokenizer(arr):
    '''
    Returns a tokenized version of input array, used in Count Vectorizer
    '''
    return (arr[0]+" "+arr[1]).split()


