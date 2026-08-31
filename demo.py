import streamlit as st
import requests
import os

st.title("Amazon Review-Based Product Recommender")

st.caption(
    "Results are shown with the closest matches first. "
    "Strong match = very similar reviews, Good match = related, "
    "Similar = a looser connection."
)


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

search_query = st.text_input("Search for a product type (e.g. 'aloe vera', 'matcha', 'dog treats')")
k = st.slider("Number of results", 1, 10, 5)

if st.button("Search") and search_query:
    response = requests.get(f"{API_URL}/search", params={"query": search_query, "k": k})

    if response.status_code == 200:
        data = response.json()
        st.write(f"**Results for:** {data['query']}")
        for rec in data["results"]:
            st.markdown(f"**{rec['product_id']}** — {rec['match_quality']}")
            st.caption(rec['title'])
            st.write(rec["sample_review"])
            st.divider()
    else:
        st.error("No results found")