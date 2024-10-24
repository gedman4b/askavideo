import streamlit as st
from openai import OpenAI
import transcript_RAG as tr

# Show title and description.
st.title("💬 Ask Your Youtube Video")
st.write(
    "This is a simple chatbot that you load a youtube url and then ask it a question about the video."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
url = st.text_input("Youtube URL")

transcript = tr.load_transcript(url)
docs_split = tr.split_text(transcript)
retriever = tr.embed_retrieve_docs(docs_split)

if not UnicodeTranslateError:
     st.info("Please enter a Youtube URL to continue.")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("Ask your question"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        """stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )"""
        convo_qa_chain.invoke(
    {
        "input": "What are autonomous agents?",
        "chat_history": [],
    }
)
        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
