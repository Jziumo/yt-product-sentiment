from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, Features, Value, ClassLabel, Dataset, DatasetDict
import evaluate
from data_process import load_df
from transformers import DataCollatorWithPadding
import os
from writer import output_to_file
from sklearn.metrics import accuracy_score, f1_score
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split

class ModelFineTune: 
    """
    Use the manually labeled data to fine-tune the model. 
    """

    def __init__(self, max_length=128):
        self.model_name="bert-base-uncased"
        self.labels = ["negative", "neutral", "positive"]
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = max_length
        self.metric = evaluate.load("accuracy")
        self.columns = ["sentiment_for_product", "sentiment_for_video"]
        # self.dataset_dict = self.load_and_cast_dataset()
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def load_and_cast_dataset(self, column_focused):
        df = load_df('combined_text_clean.csv')
        # features = Features({
        #     "text": Value("string"),
        #     "sentiment_for_product": ClassLabel(names=self.labels),
        #     "sentiment_for_video": ClassLabel(names=self.labels),
        # })
        # dataset = Dataset.from_pandas(df, features=features)

        # # Split the data into train, test and validation. 
        # train_test_split = dataset.train_test_split(test_size=0.2, seed=114)
        # test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=514) 

        # dataset_dict = DatasetDict({
        #     'train': train_test_split['train'],
        #     'test': test_val_split['train'],
        #     'validation': test_val_split['test']
        # })

        df_train, df_temp = train_test_split(
            df,
            test_size=0.2,
            stratify=df[column_focused],
            random_state=114
        )
        df_val, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp[column_focused],
            random_state=514
        )

        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
            "validation": Dataset.from_pandas(df_val.reset_index(drop=True)),
            "test": Dataset.from_pandas(df_test.reset_index(drop=True)),
        })
        
        return dataset_dict

    def preprocess(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }

    def train_single_model(self, column_focused):
        dataset_dict = self.load_and_cast_dataset(column_focused)
        # Tokenize
        tokenized = dataset_dict.map(self.preprocess, batched=True)

        # Rename label column
        tokenized = tokenized.rename_column(column_focused, "labels")
        tokenized = tokenized.remove_columns([
            col for col in ["sentiment_for_product", "sentiment_for_video", "text"] if col != column_focused
        ])

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )

        for name, param in model.base_model.named_parameters():
            if any(layer in name for layer in ["encoder.layer.10", "encoder.layer.11", "pooler"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        training_args = TrainingArguments(
            output_dir=os.path.join('./models', column_focused),
            eval_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50,
            save_strategy="epoch",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=1e-5,
            load_best_model_at_end=True,
            save_total_limit=2,
            metric_for_best_model="eval_f1", 
            warmup_steps=100,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=self.data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()

        # apply model to validation dataset
        predictions = trainer.predict(tokenized["test"])

        # Extract the logits and labels from the predictions object
        logits = predictions.predictions
        labels = predictions.label_ids

        # Use your compute_metrics function
        metrics = self.compute_metrics((logits, labels))
        
        output_to_file(filename=column_focused+'_metrics.txt', text=str(metrics))

        return model
    


if __name__ == '__main__': 
    obj = ModelFineTune()
    obj.train_single_model('sentiment_for_product')
    obj.train_single_model('sentiment_for_video')