import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def load_data(database_filepath):
    
    '''
    INPUT
    database_filepath - string for location of db file
    
    OUTPUT
    X - independent variables for input into ML model
    Y - target variable
    category_names - message category names
    '''    
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    df = pd.read_sql_table('Disaster_Resp',engine)
 
    # Sample size for testing
#    df=df.head(200).copy()
    
    X = df['message']
    Y = df[df.columns[3:]].copy()
    category_names = list(Y.columns)
    
    # Ensuring the loaded data is the same as the saved data from process_data.py
    # print(Y.info())
    # print(Y.isna().any())
    
    return X, Y, category_names


def tokenize(text):
    '''
    INPUT
    text - a string of text
    
    OUTPUT
    text - a string of text that has been processed with the steps below
        1) Normalize
        2) Remove punctuation
        3) Tokenize
        4) Lemmatize
        5) Remove stop words
    '''
    
    # lower case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]'," ",text.lower())
    
    # tokenize text to words
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    
    tokens= [ w for w in tokens if w not in stopwords.words("english")]

    return tokens

def build_model():
    '''
    INPUT
    None
    
    OUTPUT
    model - A model object built using a pipeline and GridSearch
    '''    
    
    # Text processing and model within pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('rf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Parameters for chosen classifier
    parameters = {
        'rf__estimator__n_estimators': [10, 30],
        'rf__estimator__max_depth': [5, 8],
        }

    # Instantiate GridSearchCV object based on RF parameters
    cv = GridSearchCV(pipeline,param_grid=parameters,verbose=1,cv=3)    
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    INPUT
    model - model object from build_model 
    X_test - feature data for evaluating model performance
    Y_test - target data for evaluating model performance
    category_names - message category names from load_data 
    
    OUTPUT
    None 
    '''        

    y_pred=model.predict(X_test)
    

    # Print out model performance metrics of categories
    print(classification_report(Y_test, y_pred,
                                target_names=category_names))
    
    
    for i,col in enumerate(Y_test):
        
        label=category_names[i]
        print(f'Acurracy of {label}: ',accuracy_score(Y_test[col],y_pred[:,i]))
    
    pass


def save_model(model, model_filepath):
    '''
    INPUT
    model - model object from build_model 
    model_filepath - string for directory to save model object

    OUTPUT
    None 
    '''    
    
    with open(f'{model_filepath}','wb') as f:
        pickle.dump(model,f)
    
    pass

# def load_model(model_filepath)
#   model=joblib.method?(model_filepath) 
#   return model

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
    # database_filepath=r'C:\Users\E082499\Documents\Udacity Data Scientist Nanodegree\Project 2 Disaster Pipeline\disaster_response_pipeline_project\data/DisasterResponse.db'
    # model_filepath='models/classifier.pkl'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
    

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
    
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
    
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()