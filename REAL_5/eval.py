import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ner_model import NERModel
from chunking_model import ChunkingModel

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, '='))
    print("="*80 + "\n")

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot confusion matrix using matplotlib and seaborn
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def process_ner_results(results):
    """Process and display NER evaluation results"""
    print_section_header("NAMED ENTITY RECOGNITION RESULTS")
    
    print("Classification Report:")
    print(results['report'])
    
    # Display confusion matrix
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    
    # Convert to pandas DataFrame for better display
    entity_types = sorted(cm.keys())
    cm_df = pd.DataFrame(index=entity_types, columns=entity_types)
    
    for true_entity in entity_types:
        for pred_entity in entity_types:
            cm_df.loc[true_entity, pred_entity] = cm.get(true_entity, {}).get(pred_entity, 0)
    
    print(cm_df)
    
    # Calculate overall metrics
    true_labels = results['true_labels']
    pred_labels = results['pred_labels']
    
    # Count entities (not just BIO tags)
    entity_counts = {'true': 0, 'pred': 0, 'correct': 0}
    
    # Process entities in sequence
    i = 0
    while i < len(true_labels):
        # Process true entities
        if true_labels[i].startswith('B-'):
            entity_counts['true'] += 1
            entity_type = true_labels[i].split('-')[1]
            entity_end = i
            
            # Find the end of this entity
            while entity_end + 1 < len(true_labels) and true_labels[entity_end + 1].startswith('I-') and true_labels[entity_end + 1].split('-')[1] == entity_type:
                entity_end += 1
            
            # Check if there is a matching predicted entity
            if pred_labels[i].startswith('B-') and pred_labels[i].split('-')[1] == entity_type:
                pred_match = True
                for j in range(i + 1, entity_end + 1):
                    if not (pred_labels[j].startswith('I-') and pred_labels[j].split('-')[1] == entity_type):
                        pred_match = False
                        break
                        
                if pred_match:
                    entity_counts['correct'] += 1
            
            i = entity_end + 1
        else:
            i += 1
    
    # Process predicted entities (for precision calculation)
    i = 0
    while i < len(pred_labels):
        if pred_labels[i].startswith('B-'):
            entity_counts['pred'] += 1
            i += 1
        else:
            i += 1
    
    # Calculate precision, recall, F1
    precision = entity_counts['correct'] / entity_counts['pred'] if entity_counts['pred'] > 0 else 0
    recall = entity_counts['correct'] / entity_counts['true'] if entity_counts['true'] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nEntity-level Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Entities in gold standard: {entity_counts['true']}")
    print(f"Entities in prediction: {entity_counts['pred']}")
    print(f"Correctly predicted entities: {entity_counts['correct']}")

def process_chunking_results(results):
    """Process and display Chunking evaluation results"""
    print_section_header("CHUNKING RESULTS")
    
    print("Classification Report:")
    print(results['report'])
    
    # Display confusion matrix
    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    
    # Convert to pandas DataFrame for better display
    chunk_types = sorted(cm.keys())
    cm_df = pd.DataFrame(index=chunk_types, columns=chunk_types)
    
    for true_chunk in chunk_types:
        for pred_chunk in chunk_types:
            cm_df.loc[true_chunk, pred_chunk] = cm.get(true_chunk, {}).get(pred_chunk, 0)
    
    print(cm_df)
    
    # Display per-chunk metrics
    print("\nPer-Chunk Type Metrics:")
    chunk_metrics = results['chunk_metrics']
    metrics_df = pd.DataFrame(chunk_metrics).transpose()
    metrics_df = metrics_df.sort_index()
    pd.set_option('display.precision', 4)
    print(metrics_df)
    
    # Calculate overall metrics
    true_labels = results['true_labels']
    pred_labels = results['pred_labels']
    
    # Count chunks (not just BIO tags)
    chunk_counts = {'true': 0, 'pred': 0, 'correct': 0}
    
    # Process chunks in sequence
    i = 0
    while i < len(true_labels):
        # Process true chunks
        if true_labels[i].startswith('B-'):
            chunk_counts['true'] += 1
            chunk_type = true_labels[i].split('-')[1]
            chunk_end = i
            
            # Find the end of this chunk
            while chunk_end + 1 < len(true_labels) and true_labels[chunk_end + 1].startswith('I-') and true_labels[chunk_end + 1].split('-')[1] == chunk_type:
                chunk_end += 1
            
            # Check if there is a matching predicted chunk
            if pred_labels[i].startswith('B-') and pred_labels[i].split('-')[1] == chunk_type:
                pred_match = True
                for j in range(i + 1, chunk_end + 1):
                    if not (pred_labels[j].startswith('I-') and pred_labels[j].split('-')[1] == chunk_type):
                        pred_match = False
                        break
                        
                if pred_match:
                    chunk_counts['correct'] += 1
            
            i = chunk_end + 1
        else:
            i += 1
    
    # Process predicted chunks (for precision calculation)
    i = 0
    while i < len(pred_labels):
        if pred_labels[i].startswith('B-'):
            chunk_counts['pred'] += 1
            i += 1
        else:
            i += 1
    
    # Calculate precision, recall, F1
    precision = chunk_counts['correct'] / chunk_counts['pred'] if chunk_counts['pred'] > 0 else 0
    recall = chunk_counts['correct'] / chunk_counts['true'] if chunk_counts['true'] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nChunk-level Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Chunks in gold standard: {chunk_counts['true']}")
    print(f"Chunks in prediction: {chunk_counts['pred']}")
    print(f"Correctly predicted chunks: {chunk_counts['correct']}")

def main():
    print_section_header("EVALUATION OF NER AND CHUNKING MODELS")
    
    # Define file paths
    ner_train_file = "train.csv"
    ner_test_file = "ner_annotations.txt"
    chunking_train_file = "train.txt"
    chunking_test_file = "chunk_annotations.txt"
    
    # Check if files exist
    for file_path in [ner_train_file, ner_test_file, chunking_train_file, chunking_test_file]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return
    
    print("All required files found. Starting evaluation...")
    
    # Part 1: Named Entity Recognition
    print("\nTraining NER model...")
    ner_model = NERModel()
    ner_model.train(ner_train_file)
    
    print("\nEvaluating NER model on test set...")
    ner_results = ner_model.evaluate(ner_test_file)
    process_ner_results(ner_results)
    
    # Part 2: Chunking
    print("\nTraining Chunking model...")
    chunking_model = ChunkingModel()
    chunking_model.train(chunking_train_file)
    
    print("\nEvaluating Chunking model on test set...")
    chunking_results = chunking_model.evaluate(chunking_test_file)
    process_chunking_results(chunking_results)
    
    print_section_header("EVALUATION COMPLETE")
    print("All evaluation metrics have been calculated and displayed.")

if __name__ == "__main__":
    main()