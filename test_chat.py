import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

MODEL_PATH = "subhan1501/DialoGPT-Agentic-RAG-Controller"

print(f"Loading your custom model from {MODEL_PATH}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, tie_word_embeddings=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"\nModel loaded successfully on {device.upper()}!")
print("Start typing to chat. Type 'quit' to exit.")
print("=" * 50)

chat_history_ids = None

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['quit', 'exit', 'stop']:
        print("Ending chat. Goodbye!")
        break
        
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt').to(device)

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
       
        if bot_input_ids.shape[-1] > 200:
            bot_input_ids = bot_input_ids[:, -200:]
    else:
        bot_input_ids = new_user_input_ids
        
    attention_mask = torch.ones_like(bot_input_ids)
        
    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_new_tokens=60,            
        pad_token_id=tokenizer.eos_token_id,

        eos_token_id=tokenizer.eos_token_id, 
        
        repetition_penalty=1.15,      
        no_repeat_ngram_size=3,       
        do_sample=True,               
        top_k=40,                    
        top_p=0.90,                   
        temperature=0.6       
    )
    
    bot_response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
    response = tokenizer.decode(bot_response_ids[0], skip_special_tokens=True)
    
    print(f"Bot: {response}")