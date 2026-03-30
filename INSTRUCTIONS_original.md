# SIRA: Stock Investment Research Assistant

## Design and implement a GenAI-powered assistant capable of answering user queries by combining information from:

* Unstructured data: A collection of macroeconomic and strategic documents (PDFs).
* Structured data: A CSV file containing stock information (e.g., prices, P/E ratios, market caps, etc.).

## The assistant should be able to:

* Retrieve relevant information from both unstructured and structured sources.
* Combine the retrieved information to generate coherent and accurate natural-language responses.
* Include sources of data in the response.
* All user interactions and responses should be in English.

## Data Sources

* Unstructured Data:
    * A folder containing multiple PDFs with macroeconomic and strategic insights.

* Structured Data:
    * A CSV file with columns such as: Stock Symbol, Company Name, ISIN, Sector, Stock Price, Target Price, Dividend Yield

## Functional Requirements

The assistant should be able to:
1. Answer macroeconomic or strategic questions by retrieving relevant information from the unstructured documents.
2. Answer questions about specific stocks or sectors by querying the structured CSV data.
3. Combine insights from both data sources to provide comprehensive answers.

## Example:

**User Query:** “What is the target price of Tesla, and how does it compare to the current macroeconomic trends?”  

**Expected Response:** “The target price of Tesla is 35.2. According to the latest macroeconomic report, the market is
experiencing a period of high inflation, which may impact growth stocks like Tesla.”  


## Technical Requirements

* Implement the solution in Python.
* Use a vector database for storing and querying unstructured data.
* Use a relational database for storing and querying structured data.
* Use any LLM/OCR provider (remote or local).
* Implement Text-to-SQL capabilities.
* *Do not use frameworks such as LangChain or LangGraph.

## Evaluation Criteria

* Code Quality: Clean, modular, and self-explanatory code.
* Functionality: Effective handling of all three functional requirements.
* Error Handling: Minimal but sufficient error handling for usability.
* Creativity: Robust and innovative approaches to combining structured and unstructured data.
* Performance: Efficient data processing and response generation.

## Deliverables
A Python application that includes:

* A FAST API with endpoints to:
    * Submit questions and get answers.
    * Upload new documents to be embedded.
    * Upload a new CSV to be ingested into the database.

* A README file explaining:
    * How to set up and run the assistant.
    * Any assumptions or limitations made during development.
    * Example queries the assistant can handle.