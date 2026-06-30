import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/processed/train_pairs.csv"))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../results"))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Starting Exploratory Data Analysis (EDA)...")

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"ERROR: Could not find {DATA_PATH}.")
    print("Please ensure your dataset is placed in the data/processed/ folder.")
    exit()

df = df.dropna()
df['prompt_length'] = df['prompt'].apply(lambda x: len(str(x).split()))
df['response_length'] = df['response'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 6))
sns.histplot(df['prompt_length'], color='#1f77b4', label='User Prompts', kde=True, alpha=0.6)
sns.histplot(df['response_length'], color='#ff7f0e', label='Bot Responses', kde=True, alpha=0.6)
plt.title('Distribution of Conversation Word Counts (DailyDialog)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Words', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plot_path = os.path.join(RESULTS_DIR, "eda_word_counts.png")
plt.savefig(plot_path, bbox_inches='tight', dpi=300) # High DPI for presentation slides
print(f"Saved distribution graph to: {plot_path}")

summary_path = os.path.join(RESULTS_DIR, "dataset_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Dataset Summary Statistics\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total Conversation Pairs: {len(df)}\n")
    f.write(f"Average Prompt Length: {df['prompt_length'].mean():.1f} words\n")
    f.write(f"Average Response Length: {df['response_length'].mean():.1f} words\n")
    f.write(f"Max Response Length: {df['response_length'].max()} words\n")

print(f"aved summary statistics to: {summary_path}")
print("EDA Complete!")