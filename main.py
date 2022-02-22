import argparse
import os                                     
import pandas as pd  
import time

from simpletransformers.ner import NERModel, NERArgs
import clearml
from clearml import Task
import torch
                                  
def load_text(fileName):
    f = open(fileName, 'r', encoding='utf-8')
    text = f.read()
    text = text.split('\n\n')

    df_test = pd.DataFrame()
    df_test['sentence_id'] = pd.Series()
    df_test['words'] = pd.Series()
    df_test['labels'] = pd.Series()

    sentence_num = []
    words = []
    tag = []
    for i in range(len(text)):
        sentence = text[i].split('\n')
        for j in range(len(sentence)-1):
            word = sentence[j].split(" ")
            sentence_num.append(i)
            words.append(word[0])
            tag.append(word[1])
      
    df_test['sentence_id'] = sentence_num
    df_test['words'] = words
    df_test['labels'] = tag
    return df_test

def train_model(df):
    
    model_args = NERArgs(learning_rate=1e-5, num_train_epochs=4, eval_batch_size=8, train_batch_size=8, weight_decay=1, overwrite_output_dir=True)
    model_args.labels_list = ["B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "B-PER", "I-PER", "I-DATE", "O"]
    model = NERModel(
        "xlmroberta", "xlm-roberta-large",
        args=model_args,
    )
    model.train_model(df)

    return model

def load_predict(model, fileName):
    f = open(fileName, 'r', encoding='utf-8')
    text = f.read()
    text_corpus = text.split('\n')

    predictions, raw_outputs = model.predict(text_corpus)

    return predictions

def load_best_model():
    model_args = NERArgs(learning_rate=1e-5, num_train_epochs=1, eval_batch_size=8, train_batch_size=8, weight_decay=1, overwrite_output_dir=True)
    model_args.labels_list = ["B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-DATE", "B-PER", "I-PER", "I-DATE", "O"]
    model = NERModel(
        "xlmroberta", "/content/outputs/",
        args=model_args,
    )
    return model

def new_data(df_train, predictions):
  id = len(df_train)

  keys = []
  values = []
  sentence_id = []
  for idx in range(len(predictions)):
    for length in range(len(predictions[idx])):
      for key in predictions[idx][length].keys():
        keys.append(key)
        sentence_id.append(id)
      for value in predictions[idx][length].values():
        values.append(value)

    id += 1 

  df_pred = pd.DataFrame({'sentence_id':sentence_id, 'words':keys, 'labels':values})
  df_newTrain = pd.concat([df_train, df_pred])
  df_newTrain.reset_index(inplace=True, drop=True)
  return df_newTrain

def evaluate_model(model, df_test):
    result, model_outputs, wrong_preds = model.eval_model(df_test)
    df_result = pd.DataFrame(result, index=[0])

    return df_result

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(
        description='Process the NER data and print evaluation.')
    parser.add_argument('outDir', type=str,
                        help='Output directory for the processed data')
    args = parser.parse_args()
    
    #Load training data and testing data
    train_df = load_text('train_luo.txt')
    test_df = load_text('test_luo.txt')
    train_df = train_df.head(5000)

    #train with training data - Round 1
    model = train_model(train_df)
    predictions = load_predict(model, 'luo_mat_acts.txt')

    #Create new training data but combining original + predictions
    new_train_df = new_data(train_df, predictions)
    
    #Training 75 Epochs 
    for epoch in range(75):

        print(f"Epoch {epoch+1}")
        model = load_best_model()
        model.train_model(new_train_df)
        predictions, raw_outputs = model.predict('luo_mat_acts.txt')
        new_train_df = new_data(train_df, predictions)


    #Evaluate F1 Score
    f1_score = evaluate_model(model, test_df)
    f1_score.to_csv(os.path.join(args.outDir, 'NER_results.csv'), index=False, line_terminator='\r\n')

    print('Success!')


if __name__ == "__main__":

    Task.add_requirements("-rrequirements.txt")
    task = Task.init(
    project_name='NER-Capstone',    # project name of at least 3 characters
    task_name='NER-training' + str(int(time.time())), # task name of at least 3 characters
    task_type="training",
    tags=None,
    reuse_last_task_id=True,
    continue_last_task=False,
    output_uri="s3://capstone-ner/",
    auto_connect_arg_parser=True,
    auto_connect_frameworks=True,
    auto_resource_monitoring=True,
    auto_connect_streams=True,    
    )  
    
    main()

