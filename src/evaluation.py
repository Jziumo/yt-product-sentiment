from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from writer import output_to_file
import os
import numpy as np

class Evaluation: 
    """
    Evaluate the predictions on the test data. 
    """

    def __init__(self, predictions, column_focused): 
        self.logits = predictions.predictions
        self.labels = predictions.label_ids
        self.preds = np.argmax(predictions.predictions, axis=1)
        self.column_focused = column_focused
        self.eval_res = ""
    
    def evaluate(self): 
        metrics = self.compute_metrics(self.labels, self.preds)
        cm = self.compute_confusion_matrix(self.labels, self.preds)
        classification_report = self.get_classification_report(self.labels, self.preds)

        self.eval_res = self.eval_res + str(metrics) + '\n\n'
        self.eval_res = self.eval_res + str(cm) + '\n\n'
        self.eval_res = self.eval_res + str(classification_report) + '\n\n'

        output_to_file(
            filename=self.column_focused+'_eval.txt',
            text=self.eval_res, 
            mode='w'
        )
    
        
    def compute_metrics(self, labels, preds): 
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }
    
    def compute_confusion_matrix(self, labels, preds): 
        return confusion_matrix(labels, preds)
    
    def get_classification_report(self, labels, preds): 
        return classification_report(labels, preds)