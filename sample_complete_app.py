#from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharecterTextSplitter
#from langchain_community.embeddings import Ollamaembeddings
#from langhain_community.vectorstores import FAISS
#from langchian_core.prompts import ChatPromptTemplate
#from langchain.chains.retrievel import create_retrivel_chain
#from langchain.chains.combine_document import create_stuff_document_chain
#from dotenv import load_dotenv
#import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
load_dotenv()


loader = PyPDFLoader(r"Q:\projects\ctp\critical_thinking_proj\pipeline\transformer_architecture_detailed.pdf")
document = loader.load()

test_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
final_doc = test_splitter.split_documents(document)

embeddings = OllamaEmbeddings(model =  "mxbai-embed-large:latest")
vectors = FAISS.from_documents(final_doc,embeddings)

topic = input("Enter the topic:")

llm = ChatGroq(
    model = "gemma2-9b-it",
    temperature = 0.2,
    api_key = "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"
)
prompt_question = ChatPromptTemplate.from_template("""
You are a powerful critical thinking question generator.
From the chosen topic {topic} and the provided content {context}, generate exactly 3 deep, thought-provoking questions that require higher-order thinking.

Guidelines:

Priority order for question starters:

WHY – explores causes, consequences, motivations, and implications.

WHAT – examines alternatives, scenarios, trade-offs, or challenges.

WHO – focuses on stakeholders, responsibility, or influence.
(Use HOW/WHEN/WHERE only if the context strongly supports them.)

Ensure:

At least one WHY question probing causes, effects, or relevance.

At least one WHAT question inviting alternative perspectives, solutions, or evaluations.

At least one WHO question linking to stakeholders or decision-making.

Avoid yes/no questions and simple factual recall.

Encourage analysis, evaluation, and creation (Bloom’s higher levels).

Connect at least one question to real-world application or consequences.


---

### WHO
- Who benefits from this?
- Who is this harmful to?
- Who makes decisions about this?
- Who is most directly affected?
- Who have you also heard discuss this?
- Who would be the best person to consult?
- Who will be the key people in this?
- Who deserves recognition for this?

### WHAT
- What are the strengths/weaknesses?
- What is another perspective?
- What is another alternative?
- What would be a counter-argument?
- What is the best/worst case scenario?
- What is most/least important?
- What can we do to make a positive change?
- What is getting in the way of our action?

### WHERE
- Where would we see this in the real world?
- Where are there similar concepts/situations?
- Where is there the most need for this?
- Where in the world would this be a problem?
- Where can we get more information?
- Where do we go for help with this?
- Where will this idea take us?
- Where are the areas for improvement?

### WHEN
- When is this acceptable/unacceptable?
- When would this benefit our society?
- When would this cause a problem?
- When is the best time to take action?
- When will we know we’ve succeeded?
- When has this played a part in our history?
- When can we expect this to change?
- When should we ask for help with this?

### WHY
- Why is this a problem/challenge?
- Why is it relevant to me/others?
- Why is this the best/worst scenario?
- Why are people influenced by this?
- Why should people know about this?
- Why has it been this way for so long?
- Why have we allowed this to happen?
- Why is there a need for this today?

### HOW
- How is this similar to _______?
- How does this disrupt things?
- How do we know the truth about this?
- How will we approach this safely?
- How does this benefit us/others?
- How does this harm us/others?
- How do we see this in the future?
- How can we change this for our good?

---
Output format:
Q1[Your first question]
Q2[Your second question]
Q3[Your third question]
----
begin.
""")



retriver = vectors.as_retriever()


document_chain_questions = create_stuff_documents_chain(llm,prompt_question)
retrival_question_chain = create_retrieval_chain(retriver,document_chain_questions)
response_questions = retrival_question_chain.invoke({"input":topic,"topic":topic})
questions = response_questions['answer']
print(questions)


answer_1 = input("write you answers for the question 1 :")
answer_2  = input("write you answers for the question 2 :")
answer_3 = input("write you answers for the question 3 :")

answers = answer_1 + answer_2 + answer_3
print(answers)

prompt_answers = ChatPromptTemplate.from_template("""
You are a critical thinking ability  evaluator for a critical thinking assessment.

You will be given:
- {answers}: The user's answers to generated questions.
- {questions}: The generated questions.
- {context}: Relevant reference material for evaluation.

Follow these rules :

1. Evaluation Process
   - Carefully read the user’s answers in relation to the provided questions and context.
   - Consider the depth, accuracy, relevance, and clarity of the answers.
   - Detect and count the usage of the following Bloom’s Taxonomy verbs:
     ---
     Remember verbs: Choose, Define, Find, How, Label, List, Match, Name, Omit, Recall, Relate, Select, Show, Spell, Tell, What, When, Where, Which, Who, Why
     Understand verbs: Classify, Compare, Contrast, Demonstrate, Explain, Extend, Illustrate, Infer, Interpret, Outline, Relate, Rephrase, Show, Summarize, Translate
     Apply verbs: Apply, Build, Choose, Construct, Develop, Experiment with, Identify, Interview, Make use of, Model, Organize, Plan, Select, Solve, Utilize
     Analyze verbs: Analyze, Assume, Categorize, Classify, Compare, Conclusion, Contrast, Discover, Dissect, Distinguish, Divide, Examine, Function, Inference, Inspect, List, Motive, Relationships, Simplify, Survey, Take part in, Test for, Theme
     Evaluate verbs: Agree, Appraise, Assess, Award, Choose, Compare, Conclude, Criteria, Criticize, Decide, Deduct, Defend, Determine, Disprove, Estimate, Evaluate, Explain, Importance, Influence, Interpret, Judge, Justify, Mark, Measure, Opinion, Perceive, Prioritize, Prove, Rate, Recommend, Rule on, Select, Support, Value
     Create verbs: Adapt, Build, Change, Choose, Combine, Compile, Compose, Construct, Create, Delete, Design, Develop, Discuss, Elaborate, Estimate, Formulate, Happen, Imagine, Improve, Invent, Make up, Maximize, Minimize, Modify, Original, Originate, Plan, Predict, Propose, Solution, Solve, Suppose, Test, Theory
     ---
   - The more higher-order verbs (Analyze, Evaluate, Create) the user uses appropriately, the higher the score should be.

2. Output Format 
   Your response **must** be in this exact structure without extra commentary:
   
   Evaluation of the Answers:
   - **Strengths:** [List clear strengths in bullet points]
   - **Weaknesses:** [List clear weaknesses in bullet points]
   
   Verb Usage Analysis:
   - **Detected Verbs:** [List all detected Bloom’s verbs from the user’s answers]
   - **Impact on Score:** [Explain how verb usage influenced the score]
   
   **Rating of Critical Thinking Ability:**
   RATING OF YOUR CRITICAL THINKING ABILITY IN THE SCALE OF 1 TO 10 : " mark value /  10"

3. **Scoring Guidelines**:
   - Base score on conceptual understanding, logical reasoning, clarity, and alignment with the provided context.
   - Adjust upwards if the user uses higher-order Bloom’s verbs effectively.
   - Penalize for incorrect facts, vague explanations, or lack of connection to context.
   - provide different values of marks for every iteration .

---

Now evaluate the given {answers} for the {questions} using the {context} and provide your response in the specified format only.

""")
document_chain_answer = create_stuff_documents_chain(llm,prompt_answers)
retrival_answer_chain = create_retrieval_chain(retriver,document_chain_answer)
evaluation = retrival_answer_chain.invoke({
    "input":topic,
    "topic":topic,
    "questions":questions,
    "answers":answers,
})
mark = evaluation['answer']
print("your critical thinking ability be .....")
print(mark)
