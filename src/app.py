from collections import defaultdict
import os
import streamlit as st
import voyageai
from pinecone import Pinecone
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder


# Page Config
st.set_page_config(page_title="JP Wiki RAG", page_icon="㊙️")
st.title("🇯🇵 Japanese Wiki RAG")

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

verbose = st.checkbox('Verbose Output')

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
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": english_query},
        ],
        temperature=0.0,  # Deterministic output
    )

    japanese_translation = response.choices[0].message.content.strip()

    if verbose: 
        st.write(f"Original Query: {english_query}")
        st.write(f"Japanese Translation: {japanese_translation}")
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

# Adapted from Harish Neel's public RAG tutorial series
# Source: https://github.com/harishneel1/rag-for-beginners/blob/main/11_reciprocal_rank_fusion.py
def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.metadata['text'] 
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )    
    return sorted_chunks

#note: I want to keep this seperate from the translation portion. To test each independantly. 
def multi_query_rff(user_english_query: str, alpha: float = 0.35, top_k = 5):
    prompt = """Generate 3 different versions of this query that would help retrieve relevant documents.
    Return 3 alternative queries that rephrase or approach the same question from different angles.
    Seperate with a new line and do not include any explanations. Just the queries.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_english_query},
        ],
        temperature=0.5,
    )
    alternative_queries = [user_english_query] + response.choices[0].message.content.strip().split("\n\n")
    all_results = [query_pinecone(q, alpha=alpha, top_k=top_k).matches for q in alternative_queries]
    # Apply RRF to our retrieval results
    fused_results = reciprocal_rank_fusion(all_results, k=60, verbose=True)
    return fused_results

def display_data(m): 
    string = f"""
    Japanese Title: {m.get('title_ja', 'Unknown')}
    English Title: {m.get('title_en', 'Unknown')}
    Characters: {m.get('characters', 'Unknown')}
    Year: {m.get('year', 'Unknown')}
    Author: {m.get('author', 'Unknown')}
    Content: {m.get('text', 'Unknown')}
    """
    return string


# 2. UI Input
query = st.text_input("Ask a question about Anime/Manga:", "What kind of vehicle is Hermes from Kino's Journey?")


if st.button("Search & Answer"):
    with st.spinner("Thinking..."):
        # Retrieval
        results = multi_query_rff(query, alpha=.35, top_k= 5)
        metadatas = [results[i][0].metadata for i in range(len(results))]
        i = 0
        # Context Construction
        context_list = [display_data(m) for m in metadatas]
        context_text = "\n\n---\n\n".join(context_list)
        
        # Generation
        prompt = f"""
        Answer in English. You are a helpful assistant. Answer the question based ONLY on the context provided below. If the answer isn't in the context, say you don't know.

        Context:
        {context_text}

        Question: {query}
        Answer:"""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        st.subheader("Final Answer")
        st.write(response.choices[0].message.content)
        breakpoint = 0
        with st.expander("See Retrieved Sources"):
            for result in results:
                score = result[0].score
                metadata = result[0].metadata
                st.write(f"**Score:** {score:.4f} | **Source:** {metadata.get('source', 'Unknown')}")
                st.write(f"**url:** {metadata.get('url', 'Unknown')}")
                st.info(metadata['text'][:300] + "...")