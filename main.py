import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF

# NEW: Hugging Face SentenceTransformer
from sentence_transformers import SentenceTransformer, util

# Load the transformer model (downloads automatically on first run)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ------------------ Helper Functions ------------------ #
def extract_text_from_pdf(uploaded_file):
    text = ""
    if uploaded_file is not None:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def keyword_match(resume_text, jd_text):
    resume_words = set(clean_text(resume_text).split())
    jd_words = set(clean_text(jd_text).split())
    matched = resume_words.intersection(jd_words)
    missing = jd_words - resume_words
    score = (len(matched) / len(jd_words)) * 100 if jd_words else 0
    return score, matched, missing

def cosine_similarity_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    return score

# NEW: AI Semantic Similarity using Transformers
def semantic_similarity_score(resume_text, jd_text):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_jd = model.encode(jd_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb_resume, emb_jd).item() * 100
    return score

# Function to generate PDF report
def generate_pdf_report(resume_text, jd_text, matched, missing, keyword_score, cosine_score, semantic_score, final_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Resume vs Job Description Match Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(0, 10, f"Keyword Match Score: {keyword_score:.2f}%", ln=True)
    pdf.cell(0, 10, f"TF-IDF Cosine Similarity Score: {cosine_score:.2f}%", ln=True)
    pdf.cell(0, 10, f"AI Semantic Similarity Score: {semantic_score:.2f}%", ln=True)
    pdf.cell(0, 10, f"Final Weighted Score: {final_score:.2f}%", ln=True)

    pdf.ln(10)
    pdf.cell(0, 10, "Matched Keywords:", ln=True)
    pdf.multi_cell(0, 10, ", ".join(matched) if matched else "None")

    pdf.ln(5)
    pdf.cell(0, 10, "Missing Keywords:", ln=True)
    pdf.multi_cell(0, 10, ", ".join(missing) if missing else "None")

    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        pdf_bytes = pdf_output.encode("latin1")
    elif isinstance(pdf_output, bytearray):
        pdf_bytes = bytes(pdf_output)
    else:
        pdf_bytes = pdf_output

    return pdf_bytes



# ------------------ Streamlit UI ------------------ #
st.title("AI-Powered Resume vs Job Description Matcher")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if resume_file and jd_file:
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    if resume_text and jd_text:
        # Keyword Match
        kw_score, matched_keywords, missing_keywords = keyword_match(resume_text, jd_text)

        # TF-IDF Cosine
        cs_score = cosine_similarity_score(resume_text, jd_text)

        # AI Semantic Similarity
        sem_score = semantic_similarity_score(resume_text, jd_text)

        # Final weighted score (30% keywords + 30% TF-IDF + 40% AI semantic)
        final_score = (kw_score * 0.3) + (cs_score * 0.3) + (sem_score * 0.4)

        st.subheader("Match Results")
        st.write(f"Keyword Match Score: {kw_score:.2f}%")
        st.write(f"TF-IDF Cosine Similarity Score: {cs_score:.2f}%")
        st.write(f"AI Semantic Similarity Score: {sem_score:.2f}%")
        st.write(f"Final Weighted Score: {final_score:.2f}%")

        # Word Cloud
        st.subheader("Word Cloud of Resume")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(resume_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Keyword Table
        st.subheader("Keyword Analysis Table")
        data = {
            "Matched Keywords": list(matched_keywords),
            "Missing Keywords": list(missing_keywords)
        }
        df = pd.DataFrame.from_dict(data, orient='index').transpose()
        st.dataframe(df)

        # Suggestions
        st.subheader("Suggestions")
        if missing_keywords:
            st.write("Add these missing terms to your resume for better matching:")
            st.write(", ".join(list(missing_keywords)[:15]) + ("..." if len(missing_keywords) > 15 else ""))
        else:
            st.success("Great! Your resume already covers most keywords in the JD.")

        # PDF Download
        st.subheader("Download Report")
        pdf_bytes = generate_pdf_report(resume_text, jd_text, matched_keywords, missing_keywords, kw_score, cs_score, sem_score, final_score)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="resume_vs_jd_report.pdf", mime="application/pdf")

    else:
        st.error("Could not extract text from one of the PDFs.")