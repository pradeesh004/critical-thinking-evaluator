#CRITICAL THINKING QUESTION GENERATOR & EVALUATOR

This project is an AI-powered Critical Thinking Assessment Tool that generates deep, thought-provoking questions based on a chosen topic and evaluates the user's answers for critical thinking ability using Bloom's Taxonomy.

#FEATURES

PDF Content Loader: Extracts content from a PDF document using PyPDFLoader.

Text Splitting: Splits large content into manageable chunks using RecursiveCharacterTextSplitter.

Vector Embeddings: Creates embeddings using Ollama for semantic search.

FAISS Vector Store: Stores embeddings for fast retrieval.

LLM-powered Question Generation: Generates 3 higher-order questions (WHY, WHAT, WHO) based on the selected topic.

User Answer Collection: Takes user responses to the generated questions.

Answer Evaluation: Evaluates answers based on Bloom’s Taxonomy verbs and logical reasoning.

Critical Thinking Score: Provides a score (1–10) with strengths, weaknesses, and verb usage analysis.

#TECH STACK

LangChain

Ollama Embeddings (mxbai-embed-large:latest)

FAISS

Groq LLM (gemma2-9b-it)

Python

dotenv

#PROJECT STRUCTURE

critical_thinking_proj/
│
├── pipeline/
│ ├── main.py # Main script
│ ├── transformer_architecture_detailed.pdf # Example PDF
│
├── .env # Environment variables (API keys)
├── README.txt # Project documentation

#INSTALLATION

Clone the Repository:
git clone https://github.com/your-username/critical-thinking-assessment.git

cd critical-thinking-assessment

Create Virtual Environment & Activate:
python -m venv venv
source venv/bin/activate (Linux/Mac)
venv\Scripts\activate (Windows)

Install Dependencies:
pip install langchain langchain-community langchain-ollama langchain-groq faiss-cpu python-dotenv

Set up .env file:
GROQ_API_KEY=your_groq_api_key

#USAGE

Run the script:
python main.py

#Steps:

Load PDF – The script loads and splits your PDF into chunks.

Enter Topic – Provide a topic related to the document.

Generate Questions – The model generates 3 critical thinking questions.

Answer Questions – Provide answers for the generated questions.

Evaluation – The system evaluates and provides:

Strengths

Weaknesses

Bloom's Verb Usage

Final Score (1–10)

EXAMPLE OUTPUT

Enter the topic: Transformer Architecture
Q1: Why is understanding transformer architecture crucial for modern NLP applications?
Q2: What are the main challenges in implementing transformer-based models at scale?
Q3: Who benefits most from advancements in transformer models?

write your answer for question 1: ...
write your answer for question 2: ...
write your answer for question 3: ...

your critical thinking ability be .....
Evaluation of the Answers:

Strengths: ...

Weaknesses: ...
Verb Usage Analysis:

Detected Verbs: ...
Impact on Score: ...
RATING OF YOUR CRITICAL THINKING ABILITY IN THE SCALE OF 1 TO 10 : 8/10

KEY FEATURES IN DETAIL

Question Generation Logic:

At least one WHY question (causes, effects, relevance)

At least one WHAT question (alternatives, perspectives)

At least one WHO question (stakeholders, responsibility)

Evaluation Metrics:

Depth of reasoning

Relevance to context

Bloom's Taxonomy verbs usage (higher-order → higher score)

FUTURE ENHANCEMENTS

Add Streamlit UI for better user experience.

Support for multiple PDFs.

Export results to PDF/Excel reports.

Integrate with OpenAI or Gemini as alternative LLMs.
