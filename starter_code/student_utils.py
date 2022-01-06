import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import functools


####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_code_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    ndc = (
        df
        .merge(ndc_code_df[['Non-proprietary Name', 'NDC_Code']], left_on='ndc_code', right_on='NDC_Code')
        .rename(columns={'Non-proprietary Name':"generic_drug_name"})
    )
#     assert ndc['ndc_code'].equals(ndc['NDC_Code'])
    ndc.drop(columns=['NDC_Code'], inplace=True)
    df = ndc.copy(deep=True)
      
    return df

#Question 4
## NOTE: This version keeps the first line of the first encounter, so that we have only one line per patient
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
        
    '''
    
    first_encounter_df = df.sort_values('patient_nbr').drop_duplicates(subset=['patient_nbr'], keep='first')
    
    return first_encounter_df


## NOTE: This version keeps the first encounter (multiple lines)
# def select_first_encounter(df):
#     '''
#     df: pandas dataframe, dataframe with all encounters
#     return:
#         - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
        
#     '''
    
#     first_encounter_list = df.groupby('patient_nbr')['encounter_id'].head(1).values
#     first_encounter_df = df[df['encounter_id'].isin(first_encounter_list)]
    
#     return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, valid = train_test_split(train, test_size=0.25, random_state=44)

    assert len(train[patient_key][train[patient_key].isin(test[patient_key])]) == 0
    assert len(train[patient_key][train[patient_key].isin(valid[patient_key])]) == 0
    assert len(valid[patient_key][valid[patient_key].isin(test[patient_key])]) == 0
    
    assert (len(train[patient_key]) + len(valid[patient_key]) + len(test[patient_key])) == df[patient_key].nunique()
    
    assert (len(train[patient_key]) + len(valid[patient_key]) + len(test[patient_key])) == len(df)
    
    return train, valid, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    dims = 10
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(key= c, vocabulary_file = vocab_file_path)
        
        tf_categorical_feature_column_dense = tf.feature_column.embedding_column(tf_categorical_feature_column, dimension=dims)
        
        output_tf_list.append(tf_categorical_feature_column_dense)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value=default_value, normalizer_fn=normalizer, dtype=tf.float64)
   
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = '?'
    s = '?'
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    return student_binary_prediction


# Count missing values for numeric and categorical features
def count_missing(df, numerical_col_list, categorical_col_list, predictor):
        print(predictor, "missing values = ", df[predictor].isnull().sum())
        for numerical_column in numerical_col_list:
            print(numerical_column, "missing values = ", df[numerical_column].isnull().sum())
        for cat_column in categorical_col_list:
            print(cat_column, "missing values = ", df[cat_column].isnull().sum())
        
            
