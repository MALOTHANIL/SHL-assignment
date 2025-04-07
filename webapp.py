import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
#loading the data  from excel sheet
df = pd.read_csv("shl_datafromat.csv") 
# prepare model and data name and type convert into numnerical format encoding 
model = SentenceTransformer('all-MiniLM-L6-v2')
assessment_texts = df['Name'] + " " + df['Type']
embeddings = model.encode(assessment_texts.tolist(), convert_to_tensor=True)

#building streamlite app 
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("SHL Assessment Recommendation System")
st.markdown("Enter a job description, job title, or keywords to find relevant SHL assessments.")

query = st.text_area("Job Description or Keywords", height=200)
if st.button("Recommend Assessments") and query.strip():
    with st.spinner("Finding best matches..."):
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, embeddings)[0]
        top_indices = np.argsort(scores.cpu().numpy())[::-1][:10]

        recommended = df.iloc[top_indices]
        recommended = recommended.copy()
        recommended["URL"] = recommended["URL"].apply(lambda x: f"[Link]({x})")

        st.subheader("Recommended Assessments:")
        st.write(recommended.to_markdown(index=False), unsafe_allow_html=True)
else:
    st.info("Enter a job description or keyword and click 'Recommend Assessments' to begin.")


