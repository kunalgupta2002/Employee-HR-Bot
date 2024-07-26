import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws import BedrockLLM
from dotenv import load_dotenv
import os

# Load environment variables if needed
load_dotenv()

# Initialize LangChain components
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="ELRG7F67IL",
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}}
)

# Adjust model_kwargs_claude based on the requirements of meta.llama3-8b-instruct-v1:0
model_kwargs_claude = {
    "temperature": 0,
    # Remove unsupported parameters
}

llm = BedrockLLM(model_id="meta.llama3-8b-instruct-v1:0", model_kwargs=model_kwargs_claude)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Define the Streamlit app
def main():
    st.title("Employee HR Bot")

    # Greeting message
    st.write("How can I assist you today?")

    # Initialize chat history as an empty list
    chat_history = []

    # Get user input question
    query = st.text_input("Ask a question:")

    if st.button("Ask"):
        if query:
            try:
                # Invoke LangChain model
                output = qa.invoke(query)
                answer = output['result']

                # Update chat history
                chat_history.append((query, answer))

                # Display the answer
                st.write(f"Answer: {answer}")

                # Optionally show source documents
                if 'source_documents' in output:
                    st.subheader("Source Documents:")
                    for doc in output['source_documents']:
                        st.write(doc)

            except Exception as e:
                st.error(f"Error occurred: {e}")

    # Display chat history
    if chat_history:
        st.subheader("Chat History:")
        for i, (q, a) in enumerate(chat_history, 1):
            st.write(f"Query {i}: {q}")
            st.write(f"Answer {i}: {a}")
            st.write("---")

if __name__ == "__main__":
    main()
