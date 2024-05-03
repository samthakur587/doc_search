import requests
import streamlit as st
url = 'http://fastapi:8000/get_text'
st.title("streaming response")
search_parameter = st.text_input("Enter search parameter: ")
context = st.text_input("Enter context:")

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
data = {
    "search_parameter": f"{search_parameter}",
    "context": f"{context}"
}

if st.button("Get Text"):
    response = requests.post(url, headers=headers, json=data)
    for line in response.iter_lines():
        if line:
            print(line)
            line = line.decode('utf-8')
            st.write(line)

