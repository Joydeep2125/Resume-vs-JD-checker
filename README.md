# 📄 Resume vs JD Checker  

A Streamlit web app that compares a **resume** with a **job description (JD)** to evaluate how well they match.  
It uses **Keyword Matching**, **Cosine Similarity**, and **Word Cloud Analysis** to generate insights.  
You can also **download a PDF report** with the results.  

---

## ✨ Features  
- 📑 Extracts text from Resume & Job Description PDFs  
- 🔑 Keyword Matching (matched & missing terms)  
- 📊 Cosine Similarity Score  
- ☁️ Word Cloud Visualization of Resume content  
- 📥 Downloadable PDF Report with insights  

---

## 🚀 Live Demo  
👉 [Click here to try the app on Streamlit](https://resume-vs-jd-checker-kmngappkhpakhlqmglqktaj.streamlit.app/)  

---

## ⚡ Run Locally  

Clone the repo and install dependencies:  

```bash
git clone https://github.com/YJoydeep2125/resume-vs-jd-checker.git
cd resume-vs-jd-checker
pip install -r requirements.txt


Run the Streamlit app:
streamlit run main.py


📂 Tech Stack

Python
Streamlit
Pandas
scikit-learn
Matplotlib & WordCloud
PyMuPDF (fitz) for PDF text extraction
FPDF for PDF report generation


📌 To-Do / Future Improvements

- Support for scanned PDFs (OCR integration)
- Advanced semantic similarity using NLP models (e.g., BERT)
- Resume formatting suggestions

