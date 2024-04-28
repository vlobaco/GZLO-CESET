from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pandas as pd

def create_model(df: pd.DataFrame, label: str, target: str = 'clean_text'):
    
    # Splitting the data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(df[target], df[label] , test_size=0.2, stratify=df[label], random_state=42)

    # Vectorizing the text data
    tfidf = TfidfVectorizer(min_df = 5, ngram_range=(1,2))
    X_train_tf = tfidf.fit_transform(X_train)

    # Training the model
    model = LinearSVC(random_state=0, tol=1e-5, dual='auto')
    model.fit(X_train_tf, Y_train)

    # Testing the model
    X_test_tf = tfidf.transform(X_test)
    Y_pred = model.predict(X_test_tf)

    return model, classification_report(Y_test, Y_pred)