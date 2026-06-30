import os
import csv
def generate_csv():
    print("Generating 100-row Evaluation Dataset...")
    dataset = [
        ("What is the weather like in Karachi today?", "tool"),
        ("Can you tell me the current time?", "tool"),
        ("Calculate 450 * 12", "tool"),
        ("What's the weather in London right now?", "tool"),
        ("What is 15 percent of 800?", "tool"),
        ("Do you have the live time?", "tool"),
        ("Tell me the weather in Islamabad.", "tool"),
        ("Can you add 1234 + 5678?", "tool"),
        ("What day is it today?", "tool"),
        ("Is it raining in Lahore?", "tool"),
        ("Divide 1000 / 25", "tool"),
        ("Give me the system time.", "tool"),
        ("How hot is it in Dubai?", "tool"),
        ("Evaluate 3.14 * 10 * 10", "tool"),
        ("What is the date today?", "tool"),
        ("Weather in Peshawar please.", "tool"),
        ("Calculate 55 + 45", "tool"),
        ("What is the current time in Pakistan?", "tool"),
        ("I need to calculate 99 * 99", "tool"),
        ("What is the weather in Multan?", "tool"),
        ("Tell me the date and time.", "tool"),
        ("Subtract 500 - 125", "tool"),
        ("Is it sunny in Quetta?", "tool"),
        ("What is 100 / 3?", "tool"),
        ("Give me the weather forecast for Lahore.", "tool"),
        ("Calculate 2^8", "tool"), # Evaluates to 2**8 if your regex allows it, or handles via math
        ("What time is it in New York?", "tool"),
        ("What is the temperature in Karachi?", "tool"),
        ("Calculate 85 * 3.5", "tool"),
        ("Weather update for Islamabad", "tool"),
        ("What is the exact time right now?", "tool"),
        ("Can you calculate 1000 - 456?", "tool"),
        ("How cold is it in London?", "tool"),

        # --- RAG INTENTS (Project Specific Knowledge Base) ---
        ("What is Retrieval-Augmented Generation?", "rag"),
        ("Explain how a FAISS vector database works.", "rag"),
        ("What is the DailyDialog dataset used for?", "rag"),
        ("How does TF-IDF compare to Transformer embeddings?", "rag"),
        ("What is the purpose of an Agentic layer?", "rag"),
        ("Define semantic similarity.", "rag"),
        ("What are the limitations of traditional chatbots?", "rag"),
        ("How do you calculate Cosine Similarity?", "rag"),
        ("What is a vector embedding?", "rag"),
        ("Why use all-MiniLM-L6-v2 for RAG?", "rag"),
        ("What is the ToolBench dataset?", "rag"),
        ("Explain the difference between generative and retrieval models.", "rag"),
        ("How does multi-turn memory work in this project?", "rag"),
        ("What is intent detection in NLP?", "rag"),
        ("Describe the architecture of DialoGPT.", "rag"),
        ("What is a semantic router?", "rag"),
        ("How do LLMs suffer from hallucination?", "rag"),
        ("What is the advantage of using a hybrid NLP system?", "rag"),
        ("What paper introduced RAG in 2020?", "rag"),
        ("How does the baseline model in this project work?", "rag"),
        ("What is the size of the DailyDialog dataset?", "rag"),
        ("Why avoid training an LLM from scratch?", "rag"),
        ("What does L2 distance mean in FAISS?", "rag"),
        ("How are external tools integrated into the chatbot?", "rag"),
        ("What is the intermediate architecture model?", "rag"),
        ("How does BLEU score measure text quality?", "rag"),
        ("What is the ROUGE metric?", "rag"),
        ("Explain why accuracy is used for tool selection evaluation.", "rag"),
        ("What is an intent detection diamond?", "rag"),
        ("How is conversation memory updated?", "rag"),
        ("What is the expected outcome of this FYP?", "rag"),
        ("Why did you mock the weather API?", "rag"),
        ("What is sentence-transformers?", "rag"),

        # --- LLM INTENTS (Chit-chat, Out of Scope, General) ---
        ("Hello there!", "llm"),
        ("How are you doing today?", "llm"),
        ("Tell me a joke about computers.", "llm"),
        ("What is your favorite color?", "llm"),
        ("Good morning!", "llm"),
        ("Who are you?", "llm"),
        ("Can you write a poem about AI?", "llm"),
        ("I am feeling tired today.", "llm"),
        ("What is the meaning of life?", "llm"),
        ("Do you like pizza?", "llm"),
        ("Goodbye!", "llm"),
        ("Thank you for your help.", "llm"),
        ("You are very smart.", "llm"),
        ("Tell me a story about a robot.", "llm"),
        ("What are your hobbies?", "llm"),
        ("Do you dream?", "llm"),
        ("What is your name?", "llm"),
        ("I had a great day today.", "llm"),
        ("Can you give me some life advice?", "llm"),
        ("What is the best movie of all time?", "llm"),
        ("Do you like playing video games?", "llm"),
        ("Sing a song for me.", "llm"),
        ("Why is the sky blue?", "llm"), # General knowledge, not in your DB
        ("Who won the last cricket world cup?", "llm"),
        ("I am hungry.", "llm"),
        ("Do you have any pets?", "llm"),
        ("What is your favorite food?", "llm"),
        ("Tell me something interesting.", "llm"),
        ("How do I bake a cake?", "llm"),
        ("Are you a human?", "llm"),
        ("Good evening!", "llm"),
        ("See you later.", "llm"),
        ("You are my favorite AI.", "llm"),
        ("What is the capital of France?", "llm")
    ]

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/processed"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_path = os.path.join(OUTPUT_DIR, "routing_test.csv")

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["query", "expected"]) # Write the Header
        for query, expected in dataset:
            writer.writerow([query, expected])

    print(f"\nSuccess! 100 test cases generated.")
    print(f"File saved to: {csv_path}")

if __name__ == "__main__":
    generate_csv()