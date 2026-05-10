import os
import streamlit as st
import voyageai
from pinecone import Pinecone
from openai import OpenAI

# Page Config
st.set_page_config(page_title="JP Wiki RAG", page_icon="🐎")
st.title("🇯🇵 Japanese Wiki RAG")

# 1. Setup Clients (Reading from HF Secrets)
# These will be configured in the HF Space Settings later
try:
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
except KeyError:
    st.error("Missing Voyage API Key. Please set it in the environment variables.")
    st.stop()

try:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("japanese-wiki-index")
except KeyError:
    st.error("Missing Pinecone API Key. Please set it in the environment variables.")
    st.stop()
try:
    oa_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
except KeyError:
    st.error("Missing OpenAI API Key. Please set it in the environment variables.")
    st.stop()

# 2. UI Input
query = st.text_input("Ask a question about Anime/Manga:", "What kind of vehicle is Hermes from Kino's Journey?")

if st.button("Search & Answer"):
    with st.spinner("Thinking..."):
        # 3. Retrieval
        query_emb = vo.embed([query], model="voyage-4-lite", input_type="query").embeddings[0]
        results = index.query(vector=query_emb, top_k=3, include_metadata=True)
        
        # 4. Context Construction
        context_list = [f"Content: {m.metadata['text']}" for m in results["matches"]]
        context_text = "\n\n---\n\n".join(context_list)
        
        # 5. Generation
        prompt = f"Use the context to answer in English.\n\nContext:\n{context_text}\n\nQuestion: {query}"
        response = oa_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # 6. Display Results
        st.subheader("Final Answer")
        st.write(response.choices[0].message.content)
        
        with st.expander("See Retrieved Sources"):
            for match in results["matches"]:
                st.write(f"**Score:** {match.score:.4f} | **Source:** {match.metadata.get('source', 'Unknown')}")
                st.info(match.metadata['text'][:300] + "...")