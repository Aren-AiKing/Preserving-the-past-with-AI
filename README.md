# Preserving-the-past-with-AI
# Advanced Armenian Newspaper Analysis 📰🤖

This project is an AI-powered system for automatic analysis of scanned Armenian newspapers.
It combines Computer Vision, OCR, and Natural Language Processing to extract, clean,
analyze, and summarize newspaper content.

## 🔍 What the system does

- Detects newspaper sections using YOLO
- Automatically corrects image rotation
- Evaluates image quality (blur, brightness, contrast, entropy)
- Extracts Armenian text using OCR (Tesseract)
- Cleans OCR text using AI (GPT)
- Detects article topics
- Generates Armenian summaries
- Exports results as DOCX, TXT, JSON, and ZIP files

## 🧠 Technologies Used

- Python
- Streamlit
- YOLO (Ultralytics)
- OpenCV, NumPy
- Tesseract OCR (Armenian)
- OpenAI GPT (text cleaning, topic detection, summarization)
- PIL, scikit-image

## 🖥 Interface

The project uses a Streamlit-based web interface that allows:
- Image upload
- Real-time analysis
- Visual quality metrics
- Multilingual UI (Armenian / English)
- Downloadable analysis results

## 🎯 Purpose

This project is designed for:
- Digital archives
- Libraries
- Historical newspaper analysis
- Educational and research use

## 🚀 How to run

```bash
pip install -r requirements.txt
streamlit run app.py

