# import necessary libraries
import pandas as pd
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# to interact with the operating system and interface module
import os
import shutil

# to save and load machine learning model
import joblib 

# specifyig model file and paths
model_path = 'D:\Python\Machine Learning\Resume-categorization'
csv_path = 'D:\Python\Machine Learning\categorized_resumes.csv'
output_dir = 'D:\Python\Machine Learning\Resume-categorization'
resume_dir = 'D:\Python\Machine Learning\Resume-categorization'

# Load the trained model
model = joblib.load(model_path)
resume = os.listdir(resume_dir)

# Store results on a dataframe
result = []

stemmer = nltk.stem.porter.PorterStemmer()
# Preprocess function from the model
def preprocess(text):
    ### Convert to lower case
    text = text.lower()
    
    ### Remove integers, punctuations with the help of Regular Expression
    text = re.sub('[^a-zA-Z]',' ', text)
    # Tokenization
    text = nltk.tokenize.word_tokenize(text)
    # Removing stop words
    text = [i for i in text if not i in nltk.corpus.stopwords.words('english')]
    # Stemming
    text = [stemmer.stem(i) for i in text]
    
    return ' '.join(text)

for i in resume:
    if i.endswith('pdf'):
        resume_path = os.path.join(resume_dir, i)
        
        preprocessed_resume_text = preprocess[i]


        prediction = model.predict([preprocessed_resume_text])[0]
        category_folder = os.path.join(output_dir, prediction)
        os.makedirs(category_folder, exist_ok=True)
        new_resume_path = os.path.join(category_folder, i)
        shutil.move(resume_path, new_resume_path)
        result.append({'resume': i, 'category': prediction})

#Export the results
result_df = pd.DataFrame(result)
result_df.to_csv(csv_path, index = False)

