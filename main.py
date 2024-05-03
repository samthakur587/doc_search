import streamlit as st
import pandas as pd
from pinecone import Pinecone
from openai import OpenAI
import os, requests
from dotenv import load_dotenv
import threading

# Load .env variables
load_dotenv()

# Get Pinecone and OpenAI API keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def make_api_request(text_input, context):
    out = ''
    try:
        # Make API request
        response = requests.post(
            "http://localhost:8090/analyze",
            headers={"Content-Type": "application/json"},
            json={"search_parameter": text_input, "context": context},
        )
        for line in response.iter_lines():
            if line:
                print(line)
                line = line.decode('utf-8')
                out += line
        return out
    except requests.exceptions.RequestException as e:
        st.error(f"Error making API request: {e}")
        return None

# Function to generate embeddings
def generate_embedding(text):
    # Replace 'YOUR_API_KEY' with your OpenAI API key
    client = OpenAI()

    # Call the OpenAI API to generate embeddings
    response = client.embeddings.create(
        model="text-embedding-ada-002", input=[text], encoding_format="float"
    )

    # Return the embedding
    return response.data[0].embedding

# Main function for Streamlit app
def main():
    # Title and description
    st.title("Pinecone Search Demo")
    st.write("Enter a text below and click on 'Search' to find similar items.")

    # Text input for user
    text_input = st.text_area("Enter text:")

    # Search button
    if st.button("Search"):
        # Generate embedding for input text
        embedding = generate_embedding(text_input)

        # Initialize Pinecone client
        pc = Pinecone(api_key="d33962d7-5d87-4764-9f53-c0029968f9eb")
        pc_index = pc.Index("test-compfox")

        # Perform search
        results = pc_index.query(
            namespace="test-3rd",
            vector=embedding,
            include_metadata=True,
            top_k=10,
        )

        # Display search results
        st.write("Search Results: \n")
        st.markdown(
            """
        <style>
        .result-card {
            background-color: #f0f0f0;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .result-title {
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .result-info {
            font-size: 16px;
            color: #666;
            margin-bottom: 8px;
        }
        .result-link {
            font-size: 16px;
            color: #0078e7;
        }
        .api-response {
        font-size: 16px;
        color: #666;
        margin-top: 20px;
    }

        </style>
        """,
            unsafe_allow_html=True,
        )

        # Display each search result in a card format
        for idx, result in enumerate(results["matches"]):
            metadata = result["metadata"]

            # Create a card for each result
            with st.container():
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="result-title">{metadata["name"]}</div>'
                    f'<div class="result-info">Case Code: {metadata["case_code"]}</div>'
                    f'<div class="result-info">District: {", ".join(metadata["district"])}</div>'
                    f'<div class="result-info">Decision: {metadata["decision"]}</div>'
                    f'<div class="result-info"><a class="result-link" href="{metadata["file_link"]}" target="_blank">View PDF</a></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if "text" in metadata:
                    st.markdown("##### Text Content:")
                    with st.expander("Show Text Content"):
                        st.markdown(metadata["text"])
                    context = (
                        metadata["text"]
                        .replace("'", "")
                        .replace('"', "")
                        .replace("\n", "")
                    )
                    response_placeholder = st.empty()
                    try:
                        out = make_api_request(text_input, context)
                        if out:
                            response_placeholder.markdown(
                                f"##### Why is it relevant? :\n{out}"
                            )
                        else:
                            response_placeholder.markdown(
                                "##### Why is it relevant? :\nThinking..."
                            )
                    except Exception as e:
                        st.write(e)

if __name__ == "__main__":
    main()
