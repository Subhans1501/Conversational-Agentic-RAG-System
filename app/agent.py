import os
import torch
import faiss
import pickle
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import warnings
from tools import AgentTools
warnings.filterwarnings('ignore')
class HybridAgent:
    def __init__(self):
        print("Initializing Hybrid Agentic System...")
        self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.LLM_PATH = os.path.abspath(os.path.join(self.SCRIPT_DIR, "../models/dialogpt-finetuned"))
        self.FAISS_INDEX_PATH = os.path.abspath(os.path.join(self.SCRIPT_DIR, "../models/faiss_index/vector_db.index"))
        self.FAISS_CHUNKS_PATH = os.path.abspath(os.path.join(self.SCRIPT_DIR, "../models/faiss_index/text_chunks.pkl"))
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(self.FAISS_INDEX_PATH)
        with open(self.FAISS_CHUNKS_PATH, "rb") as f:
            self.knowledge_base = pickle.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(self.LLM_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(self.LLM_PATH, tie_word_embeddings=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.chat_history_ids = None
        print(f"Agent is online on {self.device.upper()}!")
    def get_response(self, user_input: str) -> str:
        """Processes a single user message and returns the Agent's response."""
        lower_input = user_input.lower()
        if "time" in lower_input or "date" in lower_input or "what day" in lower_input:
            self.chat_history_ids = None 
            return f"**[Time Tool]**\n\n{AgentTools.get_current_time()}"
        elif "weather in" in lower_input:
            match = re.search(r'weather in ([a-zA-Z\s]+)', lower_input)
            city = match.group(1).strip() if match else "Lahore"
            self.chat_history_ids = None
            return f"**[Weather Tool]**\n\n{AgentTools.get_weather(city)}"
        elif "calculate" in lower_input or "+" in lower_input or "*" in lower_input or "/" in lower_input:
            self.chat_history_ids = None
            return f"**[Calculator Tool]**\n\n{AgentTools.calculate(user_input)}"
        user_vector = self.embedder.encode([user_input])
        distance, faiss_index = self.index.search(user_vector, 1)
        best_distance = distance[0][0]
        if best_distance < 1.5:
            retrieved_fact = self.knowledge_base[faiss_index[0][0]]
            self.chat_history_ids = None 
            return f"**[Memory Retrieved | Distance: {best_distance:.2f}]**\n\n{retrieved_fact}"
        new_user_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt').to(self.device)
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
            if bot_input_ids.shape[-1] > 200:
                bot_input_ids = bot_input_ids[:, -200:]
        else:
            bot_input_ids = new_user_input_ids
        attention_mask=torch.ones_like(bot_input_ids)
        self.chat_history_ids=self.model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=60,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=40,
            top_p=0.90,
            temperature=0.6
        )
        bot_response_ids = self.chat_history_ids[:, bot_input_ids.shape[-1]:]
        response = self.tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)
        return response