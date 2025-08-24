import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from fpdf import FPDF

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

# Function to generate PDF report
def generate_pdf_report(resume_text, jd_text, matched, missing, keyword_score, cosine_score, final_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Resume vs Job Description Match Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(0, 10, f"Keyword Match Score: {keyword_score:.2f}%", ln=True)
    pdf.cell(0, 10, f"Cosine Similarity Score: {cosine_score:.2f}%", ln=True)
    pdf.cell(0, 10, f"Final Weighted Score: {final_score:.2f}%", ln=True)

    pdf.ln(10)
    pdf.cell(0, 10, "Matched Keywords:", ln=True)
    pdf.multi_cell(0, 10, ", ".join(matched) if matched else "None")

    pdf.ln(5)
    pdf.cell(0, 10, "Missing Keywords:", ln=True)
    pdf.multi_cell(0, 10, ", ".join(missing) if missing else "None")
    
    # Handle both fpdf and fpdf2
    pdf_bytes = pdf.output(dest="S")

    # Debug info
    st.write("PDF output type:", type(pdf_bytes))

    if isinstance(pdf_bytes, str):   # old fpdf
        pdf_bytes = pdf_bytes.encode("latin1")

    return pdf_bytes



# ------------------ Streamlit UI ------------------ #
st.title("Resume vs Job Description Matcher")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if resume_file and jd_file:
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    if resume_text and jd_text:
        # --- Keyword Match ---
        kw_score, matched_keywords, missing_keywords = keyword_match(resume_text, jd_text)

        # --- Cosine Similarity ---
        cs_score = cosine_similarity_score(resume_text, jd_text)

        # Final weighted score (50% keywords + 50% cosine similarity)
        final_score = (kw_score * 0.5) + (cs_score * 0.5)

        st.subheader("Match Results")
        st.write(f"Keyword Match Score: {kw_score:.2f}%")
        st.write(f"Cosine Similarity Score: {cs_score:.2f}%")
        st.write(f"Final Weighted Score: {final_score:.2f}%")

        # --- Visualization: Word Cloud ---
        st.subheader("Word Cloud of Resume")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(resume_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # --- Table: Matched vs Missing Keywords ---
        st.subheader("Keyword Analysis Table")
        data = {
            "Matched Keywords": list(matched_keywords),
            "Missing Keywords": list(missing_keywords)
        }
        df = pd.DataFrame.from_dict(data, orient='index').transpose()
        st.dataframe(df)

        # --- Suggestions ---
        st.subheader("Suggestions")
        if missing_keywords:
            st.write("Add these missing terms to your resume for better matching:")
            st.write(", ".join(list(missing_keywords)[:15]) + ("..." if len(missing_keywords) > 15 else ""))
        else:
            st.success("Great! Your resume already covers most keywords in the JD.")

        # --- PDF Download Button ---
        st.subheader("Download Report")
        pdf_bytes = generate_pdf_report(resume_text, jd_text, matched_keywords, missing_keywords, kw_score, cs_score, final_score)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="resume_vs_jd_report.pdf", mime="application/pdf")

    else:
        st.error("Could not extract text from one of the PDFs.")
