# Chat Parser UI

A clean, Material Design-inspired web interface for querying WhatsApp chat history using semantic search.

## Features

- Parse WhatsApp chat exports and extract Q&A pairs
- Semantic search to find relevant context for user questions
- Clean, responsive Material Design UI
- Display matching contexts with relevance scores
- Option to integrate with OpenAI for more advanced answers

## Setup

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Place your WhatsApp chat export file in the root directory and name it `_chat.txt`
4. (Optional) Set your OpenAI API key in `main.py` if you want to use OpenAI for generating answers

## Running the Application

Start the FastAPI server:

```bash
python app.py
```

Then open your browser and navigate to:

```
http://localhost:8000
```

## Usage

1. Enter your question in the search box
2. Adjust the number of results you want to see (default: 3)
3. Click "Search" to find relevant information
4. View the matching contexts and the generated answer

## Customization

- To change the target answerer, modify the `target_answerer` variable in `app.py`
- To use OpenAI for generating answers, set `use_openai=True` in the `generate_answer_with_context` function call in `app.py`
- Adjust the time threshold for Q&A pairs by modifying the `time_threshold` parameter in the `extract_qa_pairs` function

## Technologies Used

- FastAPI for the backend API
- Sentence Transformers for semantic search
- Material Design for the UI
- Vanilla JavaScript for frontend functionality 