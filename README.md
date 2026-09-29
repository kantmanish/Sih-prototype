# CIL AI Reporting Assistant (SIH 2026 prototype)

Modules: (1) Automated report generation, (2) Word cloud + topic identification, (3) AI query & response.
Plus ingestion of PDF/DOCX/XLSX/images, validation, traceability and KPIs.

## Run in VS Code (recommended for screen recording)
1. Install Python 3.10+ and VS Code. Unzip this folder, then File > Open Folder.
2. Terminal:  `python -m venv venv`
   Windows: `venv\Scripts\activate`   Mac/Linux: `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `streamlit run app.py`  (opens http://localhost:8501)
5. Click "Load sample CIL documents".

## Run in Colab
```
!pip -q install -r requirements.txt   # after uploading the folder / unzipping
!npm -q install localtunnel
!streamlit run app.py &>/content/log.txt &
!curl ipv4.icanhazip.com      # this IP is the tunnel password
!npx localtunnel --port 8501
```
Open the printed URL, paste the IP as password.

## Optional
- OCR for scanned pages: install Tesseract, pytesseract is already in requirements.
- LLM answers: set `ANTHROPIC_API_KEY` (and optionally `CLAUDE_MODEL`).
