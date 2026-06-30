import os
import csv
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.agent import HybridAgent 

warnings.filterwarnings('ignore')

def main():
    print("="*50)
    print("LARGE-SCALE ROUTING & INTENT EVALUATION")
    print("="*50 + "\n")
    test_cases = []
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed/routing_test.csv'))
    
    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                test_cases.append({"query": row["query"], "expected": row["expected"].strip().lower()})
    except FileNotFoundError:
        print(f"ERROR: Could not find dataset at {csv_path}")
        print("Please create the CSV file with 'query' and 'expected' columns.")
        return
    print(f"Loaded {len(test_cases)} test cases from CSV.")
    print("Loading Hybrid Agent...\n")
    agent = HybridAgent()
    y_true = []
    y_pred = []
    print("\nRunning Evaluation (This may take a minute...)...\n")
    for i, case in enumerate(test_cases):
        query = case["query"]
        expected = case["expected"]
        y_true.append(expected)
        response = agent.get_response(query)
        if "[Weather Tool]" in response or "[Calculator Tool]" in response or "[Time Tool]" in response:
            actual = "tool"
        elif "[Memory Retrieved" in response:
            actual = "rag"
        else:
            actual = "llm"
            
        y_pred.append(actual)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_cases)} queries...")
    labels = ["tool", "rag", "llm"]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = "\n" + "="*50 + "\n"
    report += f"FINAL ROUTING METRICS REPORT (N={len(test_cases)})\n"
    report += "="*50 + "\n"
    report += f"Accuracy:  {accuracy:.4f}\n"
    report += f"Precision: {precision:.4f}\n"
    report += f"Recall:    {recall:.4f}\n"
    report += f"F1-Score:  {f1:.4f}\n\n"
    
    report += "--- Classification Report ---\n"
    report += classification_report(y_true, y_pred, labels=labels, zero_division=0) + "\n"
    
    report += "--- Confusion Matrix ---\n"
    report += f"Labels: {labels}\n"
    report += str(cm) + "\n"
    report += "="*50
    print(report)
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results"))
    report_path = os.path.join(RESULTS_DIR, "routing_metrics_100_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved full evaluation report to: {report_path}")
if __name__ == "__main__":
    main()