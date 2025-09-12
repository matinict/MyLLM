### GenTestCasesTool: An AI Agent for BDD Test Case Generation

This repository contains a Jupyter Notebook (`TestCasesGenTool.ipynb`) that demonstrates how to create an **AI agent** for generating **Behavior-Driven Development (BDD)** test cases from a user story. This tool leverages the power of **LangChain** and **Ollama** to automate a critical part of the software QA process.

-----

### **Overview**

The core of this project is an AI agent that takes a user story as input and automatically generates a comprehensive set of test cases in the **Gherkin BDD format** (`Given`/`When`/`Then`). The agent includes logic to produce a combination of valid, invalid, and edge cases, ensuring robust test coverage.

### **Technologies Used**

  * **LangChain:** Framework for building LLM applications and orchestrating the agent.
  * **Ollama:** Used to run local, open-source language models (LLMs) like **Qwen 2.5** and **Llama 3** for on-premise inference.
  * **Jupyter Notebook:** The development environment for running the code.
  * **Python:** The primary programming language.

-----

### **Getting Started**

#### **1. Setup**

First, ensure you have **Ollama** installed and running on your local machine with the models (`qwen2.5:latest` and/or `llama3:latest`) pulled and ready.

Next, install the required Python packages by running the following cell in your Jupyter Notebook:

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

#### **2. Connect to Local LLM**

This code initializes two `ChatOllama` instances, allowing you to use different LLMs for different parts of your application if needed.

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen2.5:latest",
    temperature=0.5,
    max_tokens=1000
)

tc_llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3:latest",
    temperature=0.5,
    max_tokens=1000
)
```

#### **3. Create the `generate_test_cases` Tool**

This tool defines the core logic for the agent. It uses a `PromptTemplate` to instruct the LLM to act as a QA Automation Engineer and generate BDD test cases.

```python
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

def generate_test_cases(user_story: str) -> str:
    """Generate test cases from a user stories."""
    prompt_template = PromptTemplate.from_template(
        """
        You are a QA Automation Engineer. 
        Your task is to convert the following user story into at least 10 test cases.in Gherkin BDD style  format.
        Include combinations of valid, invalid, and edge cases & alternative scenarios.

        User Story: {user_story}    

        Format:
        Feature: 
        Scenario:
            Given 
            When 
            Then
        And 

        """
    )
    prompt = prompt_template.invoke({"user_story": user_story})
    return tc_llm.invoke(prompt)
```

#### **4. Initialize and Run the Agent**

The `initialize_agent` function from LangChain ties everything together, allowing the AI to reason and decide when to use the `generate_test_cases` tool.

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

@tool
def generate_test_cases(user_story: str) -> str:
    """Generate BDD test cases from a user story."""
    return f"Generated BDD test cases for: {user_story}"

agent = initialize_agent(
    tools=[generate_test_cases],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

user_story_input = """ 
As a user, I want to reset my password using a link sent to my email,
so that I can regain access to my account if I forgot my password.
"""
result = agent.invoke({"input": f"Create BDD Test Cases For {user_story_input}"})
print(result)
```
