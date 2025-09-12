### GenTestCasesTool: An AI Agent for BDD Test Case Generation with Gemini

This repository contains a Jupyter Notebook (`TestCasesGenToolGemini.ipynb`) that demonstrates an **AI agent** built to generate **Behavior-Driven Development (BDD)** test cases from a user story. The agent leverages **LangChain** and **Google's Gemini** models to automate a critical part of the software QA process.

-----

### **Overview**

This project's core is an AI agent that takes a user story as input and automatically generates a comprehensive set of test cases in the **Gherkin BDD format** (`Given`/`When`/`Then`). The agent includes logic to produce a combination of valid, invalid, and edge cases, ensuring robust test coverage.

### **Technologies Used**

  * **LangChain:** The framework used for orchestrating the AI agent.
  * **Google Gemini:** The LLM that powers the agent's reasoning and test case generation.
  * **Jupyter Notebook:** The development environment for running the code.
  * **Python:** The primary programming language.

-----

### **Getting Started**

#### **1. Setup**

Install the necessary Python packages by running the following cell in your Jupyter Notebook. Note that you will also need to have a `.env` file with your Google API key for the Gemini models.

```bash
# %pip install -U langchain-google-genai
# %pip install -U google-genai
# %pip install --upgrade --quiet  langchain-google-genai
# %pip install dotenv
# %pip install langchain-ollama
# %pip install --upgrade --quiet  langchain-ollama
# !pip install langchain-community
# !pip install unstructured
# !pip install pdfminer
# !pip install "unstructured[all-docs]"
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
