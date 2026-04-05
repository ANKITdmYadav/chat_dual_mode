from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)

questions = [
    "What is the Transformer model architecture based on?",
    "How many encoder layers does the Transformer use?",
    "What is Scaled Dot-Product Attention?",
    "What is the formula for attention function?",
    "How many attention heads does the Transformer use?",
]

ground_truths = [
    "The Transformer is based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    "The encoder is composed of a stack of N=6 identical layers.",
    "Scaled Dot-Product Attention computes dot products of queries with all keys, divides by square root of dk, and applies softmax to get weights on values.",
    "Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) * V",
    "The Transformer uses h=8 parallel attention heads.",
]

def run_evaluation(retriever, llm):
    def retrieve(question, k=2):
        docs = retriever.invoke(question)[:k]
        return [doc.page_content for doc in docs]

    def generate_answer(question, contexts):
        context_text = "\n\n".join(contexts)
        response = llm.invoke(
            f"Answer based on context below.\n\nContext: {context_text}\n\nQuestion: {question}\nAnswer:"
        )
        return response.content

    rows = []
    for question, ground_truth in zip(questions, ground_truths):
        contexts = retrieve(question, k=2)
        answer = generate_answer(question, contexts)
        rows.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "reference": ground_truth,
        })

    evaluation_dataset = Dataset.from_list(rows)

    scores = evaluate(
        evaluation_dataset,
        metrics=[
            answer_correctness,
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ],
    )

    return scores, rows
