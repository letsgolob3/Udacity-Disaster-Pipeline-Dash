import sys
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer

def load_data(database_filepath):
    pass


def tokenize(text):
    '''
    INPUT
    text - a string of text
    
    OUTPUT
    text - a string of text that has been processed
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
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass

# def load_model(model_filepath)
#   model=joblib.method?(model_filepath) 
#   return model

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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