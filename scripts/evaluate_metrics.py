import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION & SMART PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../models/dialogpt-finetuned"))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../results"))

# Ensure the results folder exists
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading Agent for Metric Evaluation...")
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    model = AutoModelForCausalLM.from_pretrained(LLM_PATH, tie_word_embeddings=False)
except Exception as e:
    print(f"ERROR loading model: {e}")
    print("Ensure your fine-tuned model is in the models/dialogpt-finetuned/ folder.")
    exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- TEST DATASET (Gold Standard) ---
test_cases = [
    {
        "prompt": "Hello, how are you doing today?",
        "reference": "I am doing well, thank you! How can I help you?"
    },
    {
        "prompt": "I would like to book a table for two.",
        "reference": "Certainly. What time would you like the reservation for?"
    },
    {
        "prompt": "What is the weather usually like in the summer?",
        "reference": "It is usually very hot and sunny during the summer."
    }
]

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

total_bleu = 0
total_rouge1 = 0
total_rougeL = 0

report_lines = []
report_lines.append("="*50)
report_lines.append("NLP METRICS EVALUATION REPORT")
report_lines.append("="*50 + "\n")

print("\nRunning Evaluation...\n")

for i, test in enumerate(test_cases):
    prompt = test["prompt"]
    reference = test["reference"]
    
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=40,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.15
    )
    bot_response = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    ref_tokens = reference.split()
    bot_tokens = bot_response.split()
    bleu_score = sentence_bleu([ref_tokens], bot_tokens, smoothing_function=smoothie)
    rouge_scores = scorer.score(reference, bot_response)
    rouge1 = rouge_scores['rouge1'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure   
    total_bleu += bleu_score
    total_rouge1 += rouge1
    total_rougeL += rougeL
    case_result = f"Test {i+1}:\nUser: {prompt}\nReference: {reference}\nBot Generated: {bot_response}\nMetrics -> BLEU: {bleu_score:.4f} | ROUGE-1: {rouge1:.4f} | ROUGE-L: {rougeL:.4f}\n"
    print(case_result)
    report_lines.append(case_result)

avg_bleu = total_bleu / len(test_cases)
avg_rouge1 = total_rouge1 / len(test_cases)
avg_rougeL = total_rougeL / len(test_cases)
summary = "\n" + "="*50 + "\n"
summary += "FINAL SYSTEM AVERAGES\n"
summary += "-" * 50 + "\n"
summary += f"Average BLEU Score:   {avg_bleu:.4f}  (Measures exact word overlap)\n"
summary += f"Average ROUGE-1 F1:   {avg_rouge1:.4f}  (Measures keyword recall)\n"
summary += f"Average ROUGE-L F1:   {avg_rougeL:.4f}  (Measures sentence structure/flow)\n"
summary += "="*50
print(summary)
report_lines.append(summary)
report_path = os.path.join(RESULTS_DIR, "evaluation_metrics_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.writelines("\n".join(report_lines))
print(f"\nSaved full evaluation report to: {report_path}")