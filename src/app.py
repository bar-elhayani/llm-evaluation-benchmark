import os

import streamlit as st
import wikipediaapi
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

PROMPT_TEMPLATE = """\
You are a football (soccer) expert. Use the following Wikipedia context to answer the question accurately and concisely.

Context: {wikipedia_summary}

Question: {question}

If the question is in Hebrew, answer in Hebrew. If in English, answer in English."""

st.title("⚽ Football Knowledge Assistant")
st.markdown("**Powered by RAG + Llama 3.3 70B**")

question = st.text_input("Ask any football question...")

if st.button("Ask") and question.strip():
    with st.spinner("Searching Wikipedia and generating answer..."):
        wiki = wikipediaapi.Wikipedia(language="en", user_agent="FootballRAG/1.0")
        page = wiki.page(question)

        if page.exists():
            page_title = page.title
            context = page.summary[:2000]
        else:
            search_term = question.split()[0] if question.split() else question
            page = wiki.page(search_term)
            page_title = page.title if page.exists() else "No page found"
            context = page.summary[:2000] if page.exists() else ""

        prompt = PROMPT_TEMPLATE.format(wikipedia_summary=context, question=question)

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content

    st.markdown(answer)

    with st.expander("📚 Source used"):
        st.markdown(f"**Wikipedia page:** {page_title}")
        st.markdown("**Context preview (first 500 chars):**")
        st.text(context[:500])
