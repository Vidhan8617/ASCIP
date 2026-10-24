
from src.rag_chatbot import answer_question

result = answer_question("Which vendors have delivery and quality issues?")

print("\nANSWER:\n")
print(result["answer"])

print("\nSTATUS:", result["status"])
print("RECORDS RETRIEVED:", len(result["sources"]))
