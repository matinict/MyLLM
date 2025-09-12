### GenTestCasesTool: An AI Agent for BDD Test Case Generation from PDFs

This repository contains a Jupyter Notebook (`TestCasesGenToolGeminiPdf.ipynb`) that demonstrates an **AI agent** capable of generating **Behavior-Driven Development (BDD)** test cases directly from a PDF file. The agent leverages **LangChain** and a **Gemini** model to automate the process of understanding a document's requirements and creating comprehensive test scenarios.

-----

### **Project Overview**

The agent is designed to act as a **QA Automation Engineer**. It reads the content of a PDF file (e.g., a requirements document), identifies the user stories or core functionalities, and then generates at least 10 detailed test cases in the **Gherkin BDD format**. These test cases include combinations of valid, invalid, and edge cases to ensure thorough testing.

### **Technologies Used**

  * **LangChain:** The framework used to orchestrate the AI agent, connect to the LLMs, and utilize tools.
  * **Google Gemini:** The powerful LLM that drives the agent's reasoning and test case generation. The project uses both `gemini-2.5-flash` and `gemini-2.0-flash` models.
  * **Unstructured:** A library used to load and parse content from PDF files.
  * **Jupyter Notebook:** The environment for running the code and demonstrating the agent's output.

-----

### **Getting Started**

#### **1. Setup**

Install the necessary Python packages by running the following cell in your Jupyter Notebook. Note that you will also need to have a `.env` file with your Google API key for the Gemini models.

```bash
%pip install -U langchain-google-genai
%pip install -U google-genai
%pip install --upgrade --quiet  langchain-google-genai
%pip install dotenv
%pip install langchain-ollama
%pip install --upgrade --quiet  langchain-ollama
!pip install langchain-community
!pip install unstructured
!pip install pdfminer
!pip install "unstructured[all-docs]"
```

#### **2. Connect to the LLM**

This code loads your API key from a `.env` file and initializes two instances of the `ChatGoogleGenerativeAI` model.

```python
from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv

load = load_dotenv('../.env')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

tc_llm = ChatGoogleGenerativeAI( 
   model="gemini-2.0-flash"
)
```

#### **3. Create the `generate_test_cases_from_pdf` Tool**

This tool is the core of the agent's functionality. It reads PDF files from a `Pdf` directory, extracts their content, and uses a `PromptTemplate` to instruct an LLM to generate the test cases.

```python
from langchain.tools import tool
from langchain.prompts import PromptTemplate
import os
from langchain_community.document_loaders import UnstructuredFileLoader

@tool
def generate_test_cases_from_pdf(pdf_path: str) -> str:
    """Reads the requirement from the PDF and generates at least 10 test case scenarios in BDD style format"""
    Docs = "Pdf"
    documents = []

    for file in os.listdir(Docs):
        filepath = os.path.join(Docs, file)
        print(filepath)
        loader = UnstructuredFileLoader(filepath)
        docs = loader.load()
        for doc in docs:
            documents.append(f"{file} :: {doc.page_content.strip()}")

    requirement_txt = "\n\n".join(documents[:3])[:1000]
    
    prompt_template = PromptTemplate.from_template(
        """
        You are a QA Automation Engineer. 
        Your task is to convert the following user story into at least 10 test cases.in Gherkin BDD style  format.
        Include combinations of valid, invalid, and edge cases & alternative scenarios.

        User Story: {requirement_txt}    

        Format:
        Feature: 
        Scenario:
            Given 
            When 
            Then
        And 

        """
    )
    prompt = prompt_template.invoke({"requirement_txt": requirement_txt})
    return llm.invoke(prompt)
```

#### **4. Initialize and Run the Agent**

The `initialize_agent` function from LangChain uses the tool to process the `requirements.pdf` file. The verbose output shows the agent's reasoning process and the final generated test cases.

```python
from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=[generate_test_cases_from_pdf],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,    
    verbose=True
)

result = agent.run("requirements.pdf")
print(result)
```
