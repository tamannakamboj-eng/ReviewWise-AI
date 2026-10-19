REVIEWWISE AI - DATASET CHATBOT PROJECT

WHAT IS INCLUDED
- app.py: Streamlit portal with a new Dataset Chatbot page
- CIAIIR.xlsx: the supplied review dataset
- reviewwise_text_model.joblib: trained sentiment classification pipeline
- model_metrics.json: held-out model evaluation metrics
- requirements.txt: required Python packages

WHAT THE DATASET CHATBOT CAN DO
- Calculate review counts and average ratings from the Excel data
- Answer questions about positive, neutral and low-rated reviews
- Show matching review records for text queries using TF-IDF similarity
- Surface examples of low-rated reviews and frequent words in available text
- Use the trained sentiment model to classify a new review if no dataset match is found

IMPORTANT LIMITATIONS
- English review text is only partially available/translated in this workbook.
- Sentiment labels are derived from ratings: 1-2 Negative, 3 Neutral, 4-5 Positive.
- The trained model's held-out accuracy is about 70.7% on the available text subset.
- The chatbot uses dataset calculations, text similarity and the classifier; it is not a general-purpose ChatGPT-style language model.
- It cannot answer about fields not present in the workbook, such as complete customer profiles or live shipment tracking.
- Treat all outputs as an academic prototype and verify results before making business decisions.

HOW TO RUN ON WINDOWS
1. Extract the ZIP file to a normal folder, such as Documents\ReviewWise_AI_Project.
2. Install Python 3.10 or newer from python.org. During installation, tick 'Add Python to PATH'.
3. Open the extracted folder in File Explorer.
4. Click the address bar, type cmd, and press Enter.
5. Run these commands one by one:

   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   python -m streamlit run app.py

6. The portal should open in your browser. If it does not, open http://localhost:8501.
7. Choose 'Dataset Chatbot' from the left menu and ask a question.

SAMPLE QUESTIONS
- How many negative reviews are there?
- What is the average rating?
- Show me reviews mentioning delivery.
- What are customers complaining about?
- Show reviews with rating 1.
- Excellent product and fast delivery, I recommend it. (tests sentiment model if no strong dataset match is found)
