import streamlit as st
import requests

API_URL = "http://localhost:8000/search"

st.set_page_config(page_title="Patent Similarity Finder", layout="wide")

# 🔍 Title and brief instruction
st.title("🔍 AI-Powered Patent Similarity Finder")
st.markdown("Upload a patent PDF to find top similar existing patents based on abstract similarity.")

# 📄 Upload and input section
uploaded_file = st.file_uploader("📄 Choose a patent PDF", type=["pdf"])
top_k = st.number_input("Number of top similar patents to display", min_value=1, max_value=10, value=5)

if st.button("Find Similar Patents") and uploaded_file:
    with st.spinner("🔎 Extracting and searching..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            params = {"top_k": top_k}
            response = requests.post(API_URL, files=files, params=params)
            response.raise_for_status()
            data = response.json()

            # ✅ Abstract display
            st.subheader("📄 Extracted Abstract")
            abstract = data.get("abstract", "")
            if abstract:
                st.markdown(
                    f"""
                    <div style='padding:15px; background-color:#f0f9f5; 
                                border-left:5px solid #2e8b57; font-size:16px;
                                line-height:1.6; border-radius:4px;'>
                        {abstract}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ No abstract extracted from the uploaded patent.")

            # ✅ Similar patents section
            st.subheader(f"🔗 Top {top_k} Similar Patents")
            results = data.get("results", [])
            if results:
                for idx, result in enumerate(results, 1):
                    with st.expander(f"Match #{idx} (Similarity Score: {result['score']:.4f})", expanded=True):
                        pub = result.get("publication", "N/A")
                        abstract_text = result.get("abstract", result.get("text", "N/A"))

                        col1, col2 = st.columns([1, 5])
                        with col1:
                            st.markdown("**📘 Publication #:**")
                        with col2:
                            st.markdown(f"<p style='font-size:15px'>{pub}</p>", unsafe_allow_html=True)

                        col1, col2 = st.columns([1, 5])
                        with col1:
                            st.markdown("**📄 Abstract:**")
                        with col2:
                            st.markdown(
                                f"""
                                <div style='max-height:250px; overflow-y:auto; padding:12px;
                                            background-color:#f7f7f9; border-radius:6px;
                                            font-size:14px; line-height:1.6;'>
                                    {abstract_text}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            else:
                st.warning("No similar patents found.")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Error: {e}")
