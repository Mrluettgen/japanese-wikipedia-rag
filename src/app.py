import os
import streamlit as st
import voyageai
from pinecone import Pinecone
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder


# Page Config
st.set_page_config(page_title="JP Wiki RAG", page_icon="㊙️")
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
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
except KeyError:
    st.error("Missing OpenAI API Key. Please set it in the environment variables.")
    st.stop()

bm25 = BM25Encoder().load("data/japanese_bm25_model.json")

def translate_and_optimize_query(english_query: str) -> str:
    """Translates English anime queries to Japanese, preserving proper nouns

    and character names for optimal database matching.
    """
    system_prompt = (
        "You are an expert translator specializing in Japanese anime, manga, and pop culture. "
        "Your task is to translate user queries from English to Japanese so they can be searched in a Japanese Wikipedia database. "
        "Rules:\n"
        "1. Maintain accurate Japanese titles for shows (e.g., 'Uma musume' -> 'ウマ娘', 'Attack on Titan' -> '進撃の巨人').\n"
        "2. Keep character names accurate in Japanese.\n"
        "3. Return ONLY the final translated search text. Do not provide explanations or extra commentary."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Highly accurate, extremely cheap, and incredibly fast
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": english_query},
        ],
        temperature=0.0,  # Deterministic output
    )

    japanese_translation = response.choices[0].message.content.strip()

    # Combine the Japanese translation with the original English query keywords
    # This gives Voyage AI and MeCab the best of both worlds to look up.
    optimized_search_string = f"{japanese_translation} {english_query}"

    return optimized_search_string

def query_pinecone(user_english_query: str, alpha: float = 0.35, top_k = 5):
    # Translate the query to Japanese
    search_string = translate_and_optimize_query(user_english_query)

    # Generate Dense Vector from Voyage (using the translated string)
    dense_response = vo.embed(
        texts=[search_string], model="voyage-4-lite", input_type="query"
    )
    query_dense = dense_response.embeddings[0]

    # Generate Sparse Vector from MeCab BM25
    query_sparse = bm25.encode_queries(search_string)

    # Scale vectors using the Alpha parameter
    scaled_dense = [v * alpha for v in query_dense]
    scaled_sparse = {
        "indices": query_sparse["indices"],
        "values": [v * (1 - alpha) for v in query_sparse["values"]],
    }

    # Query Pinecone
    response = index.query(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=top_k,
        include_metadata=True,
    )
    return response

# 2. UI Input
query = st.text_input("Ask a question about Anime/Manga:", "What kind of vehicle is Hermes from Kino's Journey?")


if st.button("Search & Answer"):
    with st.spinner("Thinking..."):
        # Retrieval
        results = query_pinecone(query, alpha=.35, top_k= 5)
        
        # Context Construction
        context_list = [f"Content: {m.metadata['text']}" for m in results["matches"]]
        context_text = "\n\n---\n\n".join(context_list)
        
        # Generation
        prompt = f"""
        Answer in English. You are a helpful assistant. Answer the question based ONLY on the context provided below. If the answer isn't in the context, say you don't know.

        Context:
        {context_text}

        Question: {query}
        Answer:"""

        # Generate Answer (Example using OpenAI - requires 'openai' library)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", # Fast and cheap for RAG
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Display Results
        st.subheader("Final Answer")
        st.write(response.choices[0].message.content)
        
        with st.expander("See Retrieved Sources"):
            for match in results["matches"]:
                st.write(f"**Score:** {match.score:.4f} | **Source:** {match.metadata.get('source', 'Unknown')}")
                st.write(f"**url:** {match.metadata.get('url', 'Unknown')}")
                st.info(match.metadata['text'][:300] + "...")