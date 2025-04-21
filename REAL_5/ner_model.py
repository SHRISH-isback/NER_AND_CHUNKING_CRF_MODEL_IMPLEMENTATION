import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn_crfsuite
from sklearn_crfsuite import metrics

class NERModel:
    def __init__(self):
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.model_trained = False
    
    def load_data(self, filepath, format_type='csv'):
        """
        Load data from file
        """
        if format_type == 'csv':
            df = pd.read_csv(filepath)
            # Group by sentence id
            grouped = df.groupby('id').agg({'tokens': list, 'pos_tags': list, 'ner_tags': list})
            return [(row['tokens'], row['pos_tags'], row['ner_tags']) for _, row in grouped.iterrows()]
        elif format_type == 'txt':
            # For custom annotated format
            sentences = []
            tokens, pos_tags, tags = [], [], []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:  # At least token and tag
                            tokens.append(parts[0])
                            tags.append(parts[-1])  # Assume last column is NER tag
                            if len(parts) >= 3:  # If POS tags are provided
                                pos_tags.append(parts[1])
                            else:
                                pos_tags.append('NONE')
                    else:
                        if tokens:  # End of sentence
                            sentences.append((tokens, pos_tags, tags))
                            tokens, pos_tags, tags = [], [], []
            
            if tokens:  # Add the last sentence if file doesn't end with an empty line
                sentences.append((tokens, pos_tags, tags))
            
            return sentences
    
    def word2features(self, sentence, index):
        """
        Extract features for a word.
        """
        word = sentence[0][index]
        postag = sentence[1][index]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        
        # Features for words that are not at the beginning of the document
        if index > 0:
            word1 = sentence[0][index-1]
            postag1 = sentence[1][index-1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True
            
        # Features for words that are not at the end of the document
        if index < len(sentence[0])-1:
            word1 = sentence[0][index+1]
            postag1 = sentence[1][index+1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
                    
        return features
    
    def extract_features(self, sentences):
        """
        Extract features for all sentences
        """
        return [[self.word2features(s, i) for i in range(len(s[0]))] for s in sentences]
    
    def get_labels(self, sentences):
        """
        Get NER tags
        """
        return [s[2] for s in sentences]
    
    def train(self, train_data, validation_split=0.2):
        """
        Train the CRF model
        """
        sentences = self.load_data(train_data)
        
        # Split into training and validation sets
        train_sentences, val_sentences = train_test_split(sentences, test_size=validation_split, random_state=42)
        
        # Extract features and labels
        X_train = self.extract_features(train_sentences)
        y_train = self.get_labels(train_sentences)
        
        X_val = self.extract_features(val_sentences)
        y_val = self.get_labels(val_sentences)
        
        # Train the model
        self.crf.fit(X_train, y_train)
        self.model_trained = True
        
        # Evaluate on validation set
        y_pred = self.crf.predict(X_val)
        print("Validation set performance:")
        print(metrics.flat_classification_report(
            [sum(y, []) for y in y_val], 
            [sum(y, []) for y in y_pred], 
            labels=sorted(set(sum(sum(y_val, []), []))),
            digits=3
        ))
        
        return self
    
    def predict(self, test_data, format_type='txt'):
        """
        Predict NER tags for test data
        """
        if not self.model_trained:
            raise ValueError("Model must be trained before prediction")
        
        sentences = self.load_data(test_data, format_type)
        X_test = self.extract_features(sentences)
        
        # Get true labels
        y_true = self.get_labels(sentences)
        
        # Predict
        y_pred = self.crf.predict(X_test)
        
        # Return predicted and actual tags along with tokens
        return [(s[0], pred, true) for s, pred, true in zip([s[0] for s in sentences], y_pred, y_true)]
    
    def evaluate(self, test_data, format_type='txt'):
        """
        Evaluate the model on test data
        """
        if not self.model_trained:
            raise ValueError("Model must be trained before evaluation")
        
        results = self.predict(test_data, format_type)
        
        # Extract tokens, true and predicted labels
        tokens = [token for sent in results for token in sent[0]]
        true_labels = [label for sent in results for label in sent[2]]
        pred_labels = [label for sent in results for label in sent[1]]
        
        # Print classification report
        report = classification_report(true_labels, pred_labels)
        
        # Calculate confusion matrix - need to handle with care for sequence labeling
        # Extract unique entity types (ignoring B-, I- prefixes)
        entity_types = sorted(set([label.split('-')[1] if '-' in label else label for label in set(true_labels) | set(pred_labels)]))
        entity_types = [et for et in entity_types if et != 'O']  # Remove 'O' if present
        
        confusion_matrix = {}
        
        for true_label in set(true_labels):
            true_entity = true_label.split('-')[1] if '-' in true_label else true_label
            if true_entity == 'O':
                continue
                
            confusion_matrix[true_entity] = {}
            
            for pred_label in set(pred_labels):
                pred_entity = pred_label.split('-')[1] if '-' in pred_label else pred_label
                if pred_entity == 'O':
                    continue
                    
                # Count when true_entity is predicted as pred_entity
                confusion_matrix[true_entity][pred_entity] = sum(1 for t, p in zip(true_labels, pred_labels) 
                                                           if ('-' in t and t.split('-')[1] == true_entity) 
                                                           and ('-' in p and p.split('-')[1] == pred_entity))
        
        return {
            'report': report,
            'confusion_matrix': confusion_matrix,
            'true_labels': true_labels,
            'pred_labels': pred_labels
        }