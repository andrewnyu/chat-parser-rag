import re
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np
import openai

# Set your OpenAI API key here
openai.api_key = 'YOUR_OPENAI_API_KEY'

def parse_chat(chat_file):
    """
    Parses a WhatsApp exported chat file into a list of messages.
    Splits the file based on message boundaries defined by a timestamp pattern.
    Each message is a dictionary with a 'timestamp', 'sender', and 'message'.
    """
    with open(chat_file, 'r', encoding='utf-8') as f:
        chat = f.read()

    # Use re.split with a lookahead to split on the start of a new message.
    # The pattern looks for a "[" followed by a date and time (with optional seconds and AM/PM).
    message_chunks = re.split(
        r'(?=\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM)\])',
        chat
    )

    messages = []
    # Updated regex: optional tilde and whitespace before the sender.
    pattern = re.compile(
        r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s'         # Date portion e.g., "1/12/25, "
        r'(\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM))\]'  # Time portion e.g., "9:41:56 PM]"
        r'\s~?\s?'                                  # Optional " ~ " marker
        r'([^:]+):\s'                               # Sender name (until colon)
        r'(.*)$',                                   # Message text (rest of line)
        re.DOTALL
    )

    for chunk in message_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        match = pattern.match(chunk)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            sender = match.group(3).strip()
            message = match.group(4).strip()
            # Try parsing with seconds, then without seconds if necessary.
            try:
                timestamp = datetime.strptime(date_str + ' ' + time_str, '%m/%d/%y %I:%M:%S %p')
            except ValueError:
                try:
                    timestamp = datetime.strptime(date_str + ' ' + time_str, '%m/%d/%y %I:%M %p')
                except ValueError:
                    timestamp = None
            messages.append({
                'timestamp': timestamp,
                'sender': sender,
                'message': message
            })
        else:
            print("Could not parse chunk:", chunk)
    return messages

def extract_qa_pairs(messages, answerer, time_threshold=300):
    """
    Extracts Q&A pairs from messages.
    
    A message ending with '?' is taken as a question and the next message from
    the specified answerer within a time threshold (in seconds) is considered the answer.
    """
    qa_pairs = []
    for i, msg in enumerate(messages):
        if msg['message'].strip().endswith('?'):
            question = msg
            answer = None
            for j in range(i + 1, len(messages)):
                next_msg = messages[j]
                if next_msg['sender'] == answerer:
                    if (next_msg['timestamp'] - question['timestamp']).total_seconds() <= time_threshold:
                        answer = next_msg
                        break
                    else:
                        # If the gap is too large, skip pairing.
                        break
            if answer:
                qa_pairs.append((question, answer))
    return qa_pairs

def build_embedding_index(qa_pairs, model):
    """
    Builds an embedding index for the Q&A pairs.
    Each pair is represented by concatenating question and answer text.
    """
    texts = [q['message'] + " " + a['message'] for q, a in qa_pairs]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return texts, embeddings

def semantic_search(query, texts, embeddings, model, top_k=3):
    """
    Performs semantic search to retrieve the top_k most relevant Q&A contexts.
    Returns a list of (context, similarity score) tuples.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Compute cosine similarity and move the tensor to CPU before converting to NumPy.
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0].cpu().numpy()
    top_results = np.argpartition(-cosine_scores, range(top_k))[0:top_k]
    top_results = sorted(top_results, key=lambda idx: cosine_scores[idx], reverse=True)
    return [(texts[idx], float(cosine_scores[idx])) for idx in top_results]

def generate_answer_with_context(query, retrieved_context, use_openai=True):
    """
    Generates a final answer using the retrieved context via a generative model.
    If use_openai is True, it uses OpenAI's API; otherwise, it simply returns the context.
    """
    context_str = "\n".join([f"- {context}" for context in retrieved_context])
    prompt = f"""You are a domain expert in sugar process operation. Using the historical Q&A context provided, answer the following question.
    
Context:
{context_str}

Question: {query}
Answer:"""
    if use_openai:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            n=1,
            stop=None,
        )
        answer = response.choices[0].text.strip()
        return answer
    else:
        return f"Retrieved context:\n{context_str}"

def main():
    # Specify the path to your WhatsApp chat export file
    chat_file = "_chat.txt"
    messages = parse_chat(chat_file)
    print(f"Parsed {len(messages)} messages from chat.")

    #print(messages) #for debugging
    
    # Define the target answerer (the chief engineer)
    target_answerer = "R Babu"  # Adjust to match the actual sender name in your chat
    qa_pairs = extract_qa_pairs(messages, answerer=target_answerer)
    print(f"Extracted {len(qa_pairs)} Q&A pairs.")
    
    # Initialize the semantic model (requires Sentence Transformers)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts, embeddings = build_embedding_index(qa_pairs, model)
    print("Built embedding index for Q&A pairs.")
    
    # Accept user query
    query = input("Enter your question: ")
    
    # Retrieve the top relevant Q&A contexts
    retrieved = semantic_search(query, texts, embeddings, model, top_k=3)
    print("\nTop relevant historical Q&A pairs:")
    for context, score in retrieved:
        print(f"Score: {score:.4f} | Context: {context}")
    
    # Generate a final answer using retrieval augmented generation (RAG)
    final_answer = generate_answer_with_context(query, [context for context, _ in retrieved], use_openai=False)
    print("\nGenerated Answer:")
    print(final_answer)

if __name__ == "__main__":
    main()

