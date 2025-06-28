from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Features, Dataset, DatasetDict, ClassLabel, Value
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding
import data_process

model_path = "google-bert/bert-base-uncased"

# Load model tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# load model with binary classification head
id2label = {-1: "Negative", 0: "Neutral", 1: "Positive"}
label2id = {"Negative": -1, "Neutral": 0, "Positive": 1}

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3, id2label=id2label, label2id=label2id,)

# freeze all base model parameters
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# unfreeze base model pooling layers
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True


def preprocess_function(df): 
    """
    Define text preprocessing
    """
    return tokenizer(df['text'], truncation=True)


df = data_process.read_labeled_data('./data/comment_labeled/labeled_yamaha_p_225_piano_review_better_music.txt', delimiter='\t')

features = Features({
    'text': Value('string'),
    'label': ClassLabel(names=["Negative", "Neutral", "Positive"]) # Names should match your id2label/label2id
})

full_dataset = Dataset.from_pandas(df, features=features)

train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)
test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42) # Split test into test and validation

dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': test_val_split['train'],      # Your 'test' set for evaluation during training
    'validation': test_val_split['test']  # Your 'validation' set for final prediction after training
})

# preprocess all datasets
tokenized_data = dataset_dict.map(preprocess_function, batched=True)

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load metrics
accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    # get predictions
    predictions, labels = eval_pred
    
    # apply softmax to get probabilities
    probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, 
                                                                 keepdims=True)
    # use probabilities of the positive class for ROC AUC
    positive_class_probs = probabilities[:, 1]
    # compute auc
    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, 
                                     references=labels)['roc_auc'],3)
    
    # predict most probable class
    predicted_classes = np.argmax(predictions, axis=1)
    # compute accuracy
    acc = np.round(accuracy.compute(predictions=predicted_classes, 
                                     references=labels)['accuracy'],3)
    
    return {"Accuracy": acc, "AUC": auc}


# hyperparameters
lr = 2e-4
batch_size = 8
num_epochs = 10

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