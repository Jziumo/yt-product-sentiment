from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Features, Dataset, DatasetDict, ClassLabel, Value
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
import data_process
import torch
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction

model_path = "google-bert/bert-base-uncased"

# Load model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def preprocess_data(examples): 
    """
    Define text preprocessing
    """
    # take a batch of texts
    text = examples["text"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()
    return encoding


df = data_process.read_labeled_data('./data/comment_labeled/labeled_Top_5_Best_Smartphones_of_2025_-_Labeled_Comments.csv')



# features = Features({
#     'text': Value('string'),
#     'sentiment_for_product': ClassLabel(names=["-1", "0", "1"]),
#     'sentiment_for_video': ClassLabel(names=["-1", "0", "1"])
# })



# Define label names
# labels = [label for label in dataset.features.keys() if label != 'text']
labels = ["negative", "neutral", "positive"]
id2label = {-1: "negative", 0: "neutral", 1: "positive"}
label2id = {"negative": -1, "neutral": 0, "positive": 1}

# id2label = {idx:label for idx, label in enumerate(labels)}
# label2id = {label:idx for idx, label in enumerate(labels)}

features = Features({
    'text': Value('string'),
    'sentiment_for_product': ClassLabel(names=labels),
    'sentiment_for_video': ClassLabel(names=labels)
})

dataset = Dataset.from_pandas(df, features=features)

train_test_split = dataset.train_test_split(test_size=0.2, seed=114)

# Split test into test and validation
test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=514) 

dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': test_val_split['train'],
    'validation': test_val_split['test']
})

# preprocess all datasets
# tokenized_data = dataset_dict.map(preprocess_function, batched=True)
encoded_dataset = dataset_dict.map(preprocess_data, batched=True)

example = encoded_dataset['train'][0]
print(example.keys())
jz = 0

# Set the format of our data to Pytorch tensors. 
encoded_dataset.set_format("torch")

# Define model
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    problem_type="multi_label_classification", 
    num_labels=3, 
    id2label=id2label, 
    label2id=label2id
)

# freeze all base model parameters
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# unfreeze base model pooling layers
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load metrics
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

# def compute_metrics(eval_pred):
#     # get predictions
#     predictions, labels = eval_pred
    
#     # apply softmax to get probabilities
#     probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, 
#                                                                  keepdims=True)
#     # use probabilities of the positive class for ROC AUC
#     positive_class_probs = probabilities[:, 1]
#     # compute auc
#     auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, 
#                                      references=labels)['roc_auc'],3)
    
#     # predict most probable class
#     predicted_classes = np.argmax(predictions, axis=1)
#     # compute accuracy
#     acc = np.round(accuracy.compute(predictions=predicted_classes, 
#                                      references=labels)['accuracy'],3)
    
#     return {"Accuracy": acc, "AUC": auc}

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

# jz = encoded_dataset['train'][0]['labels']
# print(type(jz))

# jz = encoded_dataset['train']['input_ids'][0]
# print(jz)

# jz1 = encoded_dataset['train']['input_ids'][0].unsqueeze(0)
# jz2 = encoded_dataset['train'][0]['labels'].unsqueeze(0)
# print(jz1)
# print(jz2)
# jz3 = 0

outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))
print(outputs)


# hyperparameters
lr = 2e-4
batch_size = 8
num_epochs = 10
metric_name = 'f1'

training_args = TrainingArguments(
    output_dir="bert-phishing-classifier_teacher",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model=metric_name
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# training
trainer.train()

# apply model to validation dataset
predictions = trainer.predict(tokenized_data["validation"])

# Extract the logits and labels from the predictions object
logits = predictions.predictions
labels = predictions.label_ids

# Use your compute_metrics function
metrics = compute_metrics((logits, labels))
print(metrics)

# >> {'Accuracy': 0.889, 'AUC': 0.946}