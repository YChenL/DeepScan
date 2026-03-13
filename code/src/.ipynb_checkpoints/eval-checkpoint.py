import pandas as pd
import argparse
import json
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate_accuracy(pred_file, annotation_file="test_questions.tsv"):
    """
    Calculate accuracy and F1 score of predictions, and statistics by category
    Use sklearn to compute all metrics
    
    Args:
        pred_file: Path to prediction results file (.jsonl format)
        annotation_file: Path to annotation file (.tsv format), defaults to test_questions.tsv
        
    Returns:
        accuracy: Overall accuracy
    """
    # Get absolute path of current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read annotation file using absolute path
    df = pd.read_table(os.path.join(current_dir, annotation_file))
    
    # Read prediction results
    predictions = {}
    for line in open(pred_file):
        pred = json.loads(line)
        predictions[pred['question_id']] = pred['text'].lower()
    
    # Prepare true labels and predicted labels lists
    y_true = []
    y_pred = []
    answers = set()
    preds = set()
    
    for i in range(len(df)):
        idx = df.iloc[i]['index']
        answer = df.iloc[i]['answer'].lower()
        pred = predictions.get(idx, '').lower()
        
        answers.add(answer)
        preds.add(pred)
        
        y_true.append(answer)
        y_pred.append(pred)
    
    print("\nAnswer set:", answers)
    print("Prediction set:", preds)
    
    # Calculate overall metrics
    total_precision, total_recall, total_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    total_accuracy = accuracy_score(y_true, y_pred)
    
    # Print overall results
    print(f"\nOverall results:")
    print(f"Total samples: {len(df)}, Accuracy: {total_accuracy:.4f}, Precision: {total_precision:.4f}, Recall: {total_recall:.4f}, F1: {total_f1:.4f}")
    
    # Statistics by category
    categories = df['category'].unique()
    print("\nCategory results:")
    
    for category in categories:
        category_mask = df['category'] == category
        category_y_true = [y for i, y in enumerate(y_true) if category_mask.iloc[i]]
        category_y_pred = [y for i, y in enumerate(y_pred) if category_mask.iloc[i]]
        
        # Use sklearn to compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            category_y_true, category_y_pred, average='macro', zero_division=0
        )
        accuracy = accuracy_score(category_y_true, category_y_pred)
        
        print(f"\n{category}:")
        print(f"Samples: {sum(category_mask)}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return total_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to prediction results file")
    args = parser.parse_args()

    accuracy = evaluate_accuracy(args.path)
