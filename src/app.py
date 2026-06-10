from collections import defaultdict
import os
import streamlit as st
import voyageai
from pinecone import Pinecone
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder, List


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

def translate_and_optimize_queries(english_queries):
    """Translates a batch of English anime queries to Japanese in ONE API call."""
    system_prompt = (
        "You are an expert translator specializing in Japanese anime, manga, and pop culture. "
        "Your task is to translate user queries from English to Japanese so they can be searched in a Japanese Wikipedia database. "
        "Rules:\n"
        "1. Maintain accurate Japanese titles for shows (e.g., 'Uma musume' -> 'ウマ娘').\n"
        "2. Keep character names accurate in Japanese.\n"
        "3. Your output must strictly match the structure of the input array. "
        "Translate each line cleanly, separating answers with a single newline. Do not add numbers, explanations, or commentary."
    )
    
    # Combine list into a single batched payload string
    user_payload = "\n".join(english_queries)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        temperature=0.0,
    )

    # Parse translations back out cleanly
    
    japanese_translations = [t.strip() for t in response.choices[0].message.content.strip().split("\n")]
    if verbose: 
        st.write("Alternate Queries and Japanese Translations:")
        for en, ja in zip(english_queries, japanese_translations):
            st.write(f"Original Query: {en}")
            st.write(f"Japanese Translation: {ja}")
    # Fallback guard if the LLM output length doesn't match perfectly
    if len(japanese_translations) != len(english_queries):
        japanese_translations = (japanese_translations + [""] * len(english_queries))[:len(english_queries)]
    combined_search_strings = []
    for ja, en in zip(japanese_translations, english_queries):
        # Voyage handles this combined string perfectly
        combined_search_strings.append(f"{ja} {en}")

    # Return BOTH so MeCab doesn't get poisoned by English text
    return japanese_translations, combined_search_strings


def multi_query_rff(user_english_query: str, alpha: float = 0.7, top_k=5):
    """Generates variations, splits them properly, and executes batch query retrieval."""
    prompt = """Generate 3 different versions of this query that would help retrieve relevant documents.
    Return 3 alternative queries that rephrase or approach the same question from different angles.
    Separate with a single new line (\n) and do not include any explanations. Just the queries.
    """
    # 1 API CALL: Generate query variations
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_english_query},
        ],
        temperature=0.5,
    )
    
    alt_queries = [q.strip() for q in response.choices[0].message.content.strip().split("\n") if q.strip()]
    alternative_queries = [user_english_query] + alt_queries[:3] 
    batch_responses = batch_query_pinecone(alternative_queries, alpha=alpha, top_k=top_k)
    all_results = [res.matches for res in batch_responses]
    
    return reciprocal_rank_fusion(all_results, k=60, verbose=True)

def batch_query_pinecone(english_queries: List[str], alpha: float = 0.70, top_k=5):
    """Queries Pinecone using batching while protecting MeCab from English words.
    
    Bumped alpha default to 0.50 to give dense vectors a balanced presence.
    """
    japanese_only, combined_strings = translate_and_optimize_queries(english_queries)

    dense_response = vo.embed(
        texts=combined_strings, model="voyage-4-lite", input_type="query"
    )
    all_dense_embeddings = dense_response.embeddings

    all_responses = []
    for i in range(len(english_queries)):
        query_dense = all_dense_embeddings[i]

        query_sparse = bm25.encode_queries(japanese_only[i])

        # Scale vectors
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
        all_responses.append(response)
        
    return all_responses

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
        results = multi_query_rff(query, alpha=.7, top_k= 5)
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