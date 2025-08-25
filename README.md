# 📄 Resume vs JD Checker  

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red.svg)](https://streamlit.io/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Stars](https://img.shields.io/github/stars/YourUsername/resume-vs-jd-checker?style=social)](https://github.com/YourUsername/resume-vs-jd-checker/stargazers)  
[![Forks](https://img.shields.io/github/forks/YourUsername/resume-vs-jd-checker?style=social)](https://github.com/YourUsername/resume-vs-jd-checker/network/members)  

A **Streamlit web app** that compares a **Resume** with a **Job Description (JD)** to evaluate how well they match.  
It uses **Keyword Matching**, **Cosine Similarity**, and **AI-powered Semantic Similarity** to generate insights.  
You can also **download a detailed PDF report** with the results.  

---

## ✨ Features  
- 📑 Extracts text from **Resume & Job Description PDFs**  
- 🔑 Keyword Matching (matched & missing terms)  
- 📊 **Cosine Similarity Score** (TF-IDF based)  
- 🤖 **AI Semantic Similarity** (Sentence Transformers)  
- ☁️ Word Cloud Visualization of Resume keywords  
- 📥 Downloadable **PDF Report** with insights & scores  

---

## 🚀 Live Demo  
👉 [Try the app on Streamlit](https://resume-vs-jd-checker-kmngappkhpakhlqmglqktaj.streamlit.app/)  

---

## ⚡ Run Locally  

Clone the repo and install dependencies:  

```bash
git clone https://github.com/YourUsername/resume-vs-jd-checker.git
cd resume-vs-jd-checker
pip install -r requirements.txt


Run the Streamlit app:
streamlit run main.py


📂 Tech Stack
Python
Streamlit (UI)
Pandas (data handling)
scikit-learn (TF-IDF & cosine similarity)
SentenceTransformers (semantic similarity)
Matplotlib & WordCloud (visualization)
PyMuPDF (fitz) (PDF text extraction)
FPDF (PDF report generation)



📌 Future Improvements
📷 Support for scanned PDFs (OCR integration)
🧠 Advanced semantic similarity using large NLP models (e.g., BERT)
🎨 Resume formatting & improvement suggestions