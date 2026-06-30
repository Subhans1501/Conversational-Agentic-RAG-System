import os
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
class BaselineAgent:
    def __init__(self, model_dir="models/tfidf", data_path="data/processed/train_pairs.csv"):
        print("Initializing Agent...")
        print("Loading conversation history...")
        self.df = pd.read_csv(data_path)
        self.df['prompt'] = self.df['prompt'].fillna("")
        self.df['response'] = self.df['response'].fillna("")
        print("Loading TF-IDF models...")
        with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(os.path.join(model_dir, "tfidf_matrix.pkl"), "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        print("Agent is ready!\n" + "="*40)

    def generate_response(self, user_query):
        query_vec = self.vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_match_idx = similarities.argmax()
        best_score = similarities[best_match_idx]
        if best_score < 0.15:
            return "I'm not entirely sure how to respond to that."       
        return self.df.iloc[best_match_idx]['response']
if __name__ == "__main__":
    bot = BaselineAgent()
    print("\nWelcome to the Conversational AI (Baseline Version).")
    print("Type 'quit' or 'exit' to stop the chat.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Bot: Goodbye! Have a great day.")
            break
        bot_reply = bot.generate_response(user_input)
        print(f"Bot: {bot_reply}")