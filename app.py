import json
import re
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
DATA_FILE = BASE / "CIAIIR.xlsx"
MODEL_FILE = BASE / "reviewwise_text_model.joblib"
METRICS_FILE = BASE / "model_metrics.json"

st.set_page_config(page_title="ReviewWise AI", page_icon="💬", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE, sheet_name="English_Data")
    df["Review_Creation_Date"] = pd.to_datetime(df["Review_Creation_Date"], errors="coerce")
    df["Rating_1_to_5"] = pd.to_numeric(df["Rating_1_to_5"], errors="coerce")
    for col in ["Review_Title_English", "Review_Comment_English", "Sentiment_Label_English"]:
        df[col] = df[col].fillna("").astype(str)
    df["Review_Text"] = (df["Review_Title_English"].str.strip() + " " + df["Review_Comment_English"].str.strip()).str.strip()
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_metrics():
    with open(METRICS_FILE, encoding="utf-8") as f:
        return json.load(f)

df = load_data()
model = load_model()
metrics = load_metrics()
total = len(df)
avg = df["Rating_1_to_5"].mean()
counts = df["Sentiment_Label_English"].value_counts()
neg_count = int((df["Rating_1_to_5"] <= 2).sum())
pos_count = int((df["Rating_1_to_5"] >= 4).sum())
text_rows = int(df["Review_Text"].ne("").sum())

st.sidebar.title("💬 ReviewWise AI")
st.sidebar.caption("Customer Review Intelligence Portal")
page = st.sidebar.radio("Navigate", ["Dashboard", "Dataset Chatbot", "Review Explorer", "AI Sentiment Classifier", "Model Evaluation", "About & Limitations"])
st.sidebar.divider()
st.sidebar.caption("CIA prototype • Olist review dataset")
st.title({"Dashboard":"Customer Review Dashboard", "Dataset Chatbot":"Chat with Your Dataset", "Review Explorer":"Review Explorer", "AI Sentiment Classifier":"AI Sentiment Classifier", "Model Evaluation":"Model Evaluation", "About & Limitations":"About the Prototype"}[page])
st.caption("Explore customer ratings, investigate low-rated reviews and test a text-classification model.")

if page == "Dataset Chatbot":
    st.markdown("Ask questions about the Olist review dataset. Answers are grounded in the loaded spreadsheet, with matching records shown when available.")
    st.info("Try: ‘How many negative reviews?’, ‘What is the average rating?’, ‘Show reviews mentioning delivery’, or ‘What are customers complaining about?’")

    # Build a lightweight search index from the English text actually present in the workbook.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    @st.cache_resource
    def make_search_index(texts):
        usable = [i for i, text in enumerate(texts) if str(text).strip()]
        corpus = [str(texts[i]) for i in usable]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_features=50000)
        matrix = vectorizer.fit_transform(corpus) if corpus else None
        return usable, vectorizer, matrix

    text_values = df["Review_Text"].fillna("").astype(str).tolist()
    usable_idx, search_vectorizer, search_matrix = make_search_index(text_values)

    def show_results(matched, heading="Matching review records"):
        st.subheader(heading)
        st.write(f"Found **{len(matched):,}** matching records.")
        if len(matched) == 0:
            st.write("No matching records were found in the available dataset text.")
            return
        columns = ["Review_ID", "Order_ID", "Review_Creation_Date", "Rating_1_to_5", "Sentiment_Label_English", "Review_Title_English", "Review_Comment_English"]
        out = matched[columns].head(20).copy()
        out["Review_Creation_Date"] = pd.to_datetime(out["Review_Creation_Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(out, use_container_width=True, hide_index=True)
        if len(matched) > 20:
            st.caption("Showing the first 20 results here. Use Review Explorer to filter or export a larger result set.")

    def answer_question(question):
        q = question.strip()
        qlow = q.lower()
        if not q:
            return
        st.markdown("**You:** " + q)
        # Dataset-wide numerical questions.
        if any(term in qlow for term in ["how many", "number of", "count", "total reviews", "how much data"]) and not re.search(r"(?:rating|rated|stars?|score(?:d)?)\s*(?:of|is|=|:)?\s*[1-5]\b|\b[1-5]\s*[- ]?star", qlow) and not any(term in qlow for term in ["mention", "about", "delivery", "damaged", "late", "shipping", "broken", "refund", "complain", "complaint"]):
            if "negative" in qlow or "bad review" in qlow or "low rating" in qlow:
                n = int((df["Rating_1_to_5"] <= 2).sum())
                st.markdown(f"**ReviewWise AI:** There are **{n:,} low-rated reviews** (rating 1 or 2), which is **{n/total:.1%}** of all {total:,} records.")
                show_results(df[df["Rating_1_to_5"] <= 2].sort_values("Rating_1_to_5"), "Examples of low-rated reviews")
                return
            if "positive" in qlow:
                n = int((df["Rating_1_to_5"] >= 4).sum())
                st.markdown(f"**ReviewWise AI:** **{n:,} reviews** have ratings of 4 or 5 ({n/total:.1%} of all records). These are rating-derived positive labels, not independently verified text sentiment.")
                show_results(df[df["Rating_1_to_5"] >= 4].sort_values("Rating_1_to_5", ascending=False), "Examples of positive-rated reviews")
                return
            if "neutral" in qlow:
                n = int((df["Rating_1_to_5"] == 3).sum())
                st.markdown(f"**ReviewWise AI:** **{n:,} reviews** have a rating of 3 ({n/total:.1%} of all records).")
                show_results(df[df["Rating_1_to_5"] == 3], "Examples of 3-star reviews")
                return
            st.markdown(f"**ReviewWise AI:** The dataset contains **{total:,} review records**. The average rating is **{avg:.2f}/5**. There are **{text_rows:,}** records with some English title/comment text available.")
            return
        if "average rating" in qlow or "mean rating" in qlow or "rating average" in qlow:
            st.markdown(f"**ReviewWise AI:** The average rating across **{total:,} records** is **{avg:.2f} out of 5**.")
            st.write("This is calculated directly from the Rating_1_to_5 column in the spreadsheet.")
            return
        if "rating distribution" in qlow or "ratings distribution" in qlow or "breakdown of ratings" in qlow:
            dist = df["Rating_1_to_5"].value_counts().reindex([1,2,3,4,5], fill_value=0).rename_axis("Rating").reset_index(name="Reviews")
            st.dataframe(dist, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(dist, x="Rating", y="Reviews", title="Review count by star rating"), use_container_width=True)
            return
        if any(term in qlow for term in ["complain", "complaint", "problem", "issue", "unhappy", "what are customers saying", "main concern"]):
            neg = df[(df["Rating_1_to_5"] <= 2) & df["Review_Text"].str.strip().ne("")].copy()
            if len(neg):
                words = re.findall(r"[a-zA-Z]{3,}", " ".join(neg["Review_Text"].str.lower().tolist()))
                stop = {"the","and","for","was","with","this","that","not","but","are","you","too","very","have","has","had","from","they","them","its","it","was","were","been","product","order","review","customer","delivery","good","bad","one","all","can","would","could","after","before","when","what","there","their","just","more","some","about"}
                from collections import Counter
                common = [(w,c) for w,c in Counter(words).most_common(15) if w not in stop][:8]
                st.markdown("**ReviewWise AI:** I checked low-rated reviews that contain available English text. These frequent words can provide clues, but they are not a validated topic model:")
                if common:
                    st.write(", ".join(f"**{w}** ({c})" for w,c in common))
                show_results(neg.sort_values(["Rating_1_to_5", "Review_Creation_Date"]), "Low-rated reviews with available English text")
            else:
                st.markdown("**ReviewWise AI:** I couldn't find low-rated reviews with available English text. Many original review comments are not fully translated in this file.")
            return
        # Explicit rating requests.
        rating_match = re.search(r"(?:rating|rated|stars?|score(?:d)?)\s*(?:of|is|=|:)?\s*([1-5])|([1-5])\s*[- ]?star", qlow)
        if rating_match:
            r = int(next(g for g in rating_match.groups() if g))
            matched = df[df["Rating_1_to_5"] == r]
            st.markdown(f"**ReviewWise AI:** There are **{len(matched):,} reviews** with a rating of **{r}/5** ({len(matched)/total:.1%} of all records).")
            show_results(matched, f"Examples of {r}-star reviews")
            return
        # Search for the most relevant records using TF-IDF similarity on available English review text.
        if search_matrix is not None and len(usable_idx):
            qvec = search_vectorizer.transform([q])
            sims = cosine_similarity(qvec, search_matrix).ravel()
            order = sims.argsort()[::-1]
            relevant = [(usable_idx[i], float(sims[i])) for i in order[:100] if sims[i] > 0.04]
            if relevant:
                ids = [i for i, score in relevant]
                matched = df.iloc[ids].copy()
                matched["Relevance_Score"] = [score for i, score in relevant]
                st.markdown(f"**ReviewWise AI:** I found **{len(relevant)} potentially relevant reviews** in the available English text for your question. These are text matches, not proof that every result answers the question.")
                show_results(matched, "Most relevant review records")
                return
        # If the user enters a review-like sentence, use the trained classifier as a secondary signal.
        if len(q.split()) >= 4:
            try:
                pred = model.predict([q])[0]
                probs = model.predict_proba([q])[0]
                conf = float(probs[list(model.classes_).index(pred)])
                st.markdown(f"**ReviewWise AI:** I couldn't find a strong match in the dataset text. If this is a new customer review, the trained sentiment model predicts **{pred}** (estimated probability {conf:.1%}).")
                st.caption("This is a model prediction about the text you typed, not a fact retrieved from the dataset.")
                return
            except Exception:
                pass
        st.markdown("**ReviewWise AI:** I couldn't find a reliable answer in the available columns or English review text. Try asking about review counts, ratings, dates, sentiment labels, or words such as ‘delivery’ or ‘damaged’. The spreadsheet does not contain full customer profiles, product details, or shipment tracking information.")

    with st.form("dataset_chat_form", clear_on_submit=True):
        question = st.text_input("Ask a question about the dataset", placeholder="e.g. What are customers complaining about?")
        submitted = st.form_submit_button("Ask ReviewWise AI", type="primary")
    if submitted and question.strip():
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append(question.strip())
        answer_question(question)
    st.divider()
    st.caption("How it works: dataset statistics are calculated directly from Excel; review lookup uses TF-IDF text similarity; new review text can be classified by the trained Logistic Regression model. This is a dataset-grounded academic prototype, not a general-purpose LLM.")

elif page == "Dashboard":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total reviews", f"{total:,}")
    c2.metric("Average rating", f"{avg:.2f} / 5")
    c3.metric("Positive rating labels", f"{pos_count:,}", f"{pos_count/total:.2%}")
    c4.metric("Negative rating labels", f"{neg_count:,}", f"{neg_count/total:.2%}")
    st.info(f"Priority signal: {neg_count:,} records have ratings of 1 or 2. These are flagged by a transparent rating rule, not by the text model.")
    left,right=st.columns(2)
    with left:
        sent_counts = df["Sentiment_Label_English"].value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
        fig=px.pie(sent_counts, names="Sentiment", values="Reviews", hole=.45, title="Rating-derived sentiment categories")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        monthly=df.dropna(subset=["Review_Creation_Date"]).groupby(df["Review_Creation_Date"].dt.to_period("M").astype(str))["Rating_1_to_5"].mean().reset_index()
        monthly.columns=["Month","Average rating"]
        fig=px.line(monthly,x="Month",y="Average rating",markers=True,title="Average rating by month")
        fig.update_yaxes(range=[1,5])
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Business takeaways")
    a,b,c=st.columns(3)
    a.markdown("**Most ratings are positive**\n\nAbout 77% of records are classified as positive using ratings of 4–5.")
    b.markdown("**Negative feedback needs attention**\n\nRatings of 1–2 can be flagged so staff can investigate the actual customer experience.")
    c.markdown("**Trends need context**\n\nMonthly changes show when to investigate, but do not prove why ratings changed.")
    st.caption("Labels shown on this dashboard are derived from ratings. They are not independent human-verified sentiment labels.")

elif page == "Review Explorer":
    st.write("Search and filter the supplied review records. English text may be unavailable for many rows because the source comments were not fully translated.")
    f1,f2,f3,f4=st.columns([2,1,1,1])
    with f1: query=st.text_input("Search ID or available English text")
    with f2: sentiment=st.selectbox("Rating-derived label",["All","Negative","Neutral","Positive"])
    with f3: rating=st.selectbox("Rating",["All",1,2,3,4,5])
    with f4:
        months=sorted(df["Review_Creation_Date"].dropna().dt.strftime("%Y-%m").unique())
        month=st.selectbox("Month",["All"]+months)
    view=df
    if query:
        q=query.lower()
        view=view[view["Review_ID"].astype(str).str.lower().str.contains(q,na=False) | view["Review_Text"].str.lower().str.contains(q,na=False)]
    if sentiment!="All": view=view[view["Sentiment_Label_English"]==sentiment]
    if rating!="All": view=view[view["Rating_1_to_5"]==int(rating)]
    if month!="All": view=view[view["Review_Creation_Date"].dt.strftime("%Y-%m")==month]
    st.write(f"**{len(view):,}** matching records")
    show=view[["Review_ID","Order_ID","Review_Creation_Date","Rating_1_to_5","Sentiment_Label_English","Review_Title_English","Review_Comment_English"]].copy()
    show["Review_Creation_Date"]=show["Review_Creation_Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(show.head(1000), use_container_width=True, hide_index=True)
    st.download_button("Download filtered records as CSV", data=show.to_csv(index=False).encode("utf-8-sig"), file_name="reviewwise_filtered_reviews.csv", mime="text/csv")
    if len(view)>1000: st.caption("For browser performance, the table displays the first 1,000 matching records. The CSV export includes all matching records.")

elif page == "AI Sentiment Classifier":
    st.markdown("Enter a review in English to see the model's predicted category. This is an experimental text model trained on the subset of rows with available English text.")
    sample=st.selectbox("Try an example", ["Write or select a sample", "The delivery was late and the product arrived damaged.", "Excellent product, fast delivery, I recommend it.", "The product is okay but delivery took too long."])
    default_text="" if sample=="Write or select a sample" else sample
    review=st.text_area("Review text", value=default_text, height=130, placeholder="Type a customer review here…")
    if st.button("Classify review", type="primary"):
        if not review.strip():
            st.warning("Enter some review text first.")
        else:
            pred=model.predict([review.strip()])[0]
            probs=model.predict_proba([review.strip()])[0]
            classes=model.classes_
            probability=float(probs[list(classes).index(pred)])
            colors={"Positive":"green","Neutral":"orange","Negative":"red"}
            st.subheader("Model result")
            st.markdown(f"### Predicted category: :{colors.get(pred,'blue')}[{pred}]")
            st.metric("Model confidence (estimated probability)", f"{probability:.1%}")
            prob_df=pd.DataFrame({"Category":classes,"Estimated probability":probs}).sort_values("Estimated probability",ascending=False)
            st.plotly_chart(px.bar(prob_df,x="Category",y="Estimated probability",range_y=[0,1],title="Model class probabilities"),use_container_width=True)
            st.warning("This prediction is experimental. The model learned from rating-derived proxy labels and a limited set of available English phrases. Do not use it to make real customer decisions without validation.")
    st.divider()
    st.markdown("**Model type:** TF-IDF word/bigram features + Logistic Regression with class weighting.")
    st.markdown(f"**Training records:** {metrics['training_rows']:,} · **Held-out test records:** {metrics['test_rows']:,} · **Held-out accuracy:** {metrics['accuracy']:.1%}")

elif page == "Model Evaluation":
    st.subheader("Held-out test results")
    c1,c2,c3=st.columns(3)
    c1.metric("Training text records",f"{metrics['training_rows']:,}")
    c2.metric("Test text records",f"{metrics['test_rows']:,}")
    c3.metric("Accuracy",f"{metrics['accuracy']:.1%}")
    st.write("The data was split into training and test subsets using a stratified 80/20 split. Accuracy alone is not enough, especially because the positive class is much more common.")
    report=metrics["classification_report"]
    tab=pd.DataFrame([{"Class":name,"Precision":report[name]["precision"],"Recall":report[name]["recall"],"F1-score":report[name]["f1-score"],"Support":int(report[name]["support"])} for name in ["Negative","Neutral","Positive"]])
    st.dataframe(tab.style.format({"Precision":"{:.2f}","Recall":"{:.2f}","F1-score":"{:.2f}"}),use_container_width=True,hide_index=True)
    labels=metrics["confusion_matrix_labels"]
    cm=metrics["confusion_matrix"]
    fig=px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale="Blues",labels=dict(x="Predicted label",y="Actual proxy label",color="Count"),title="Confusion matrix (rows = actual proxy label)")
    st.plotly_chart(fig,use_container_width=True)
    st.warning("Interpretation: the model's test accuracy is about 70.7%, but its results are affected by imbalanced classes, limited English text and labels derived from ratings rather than human-verified sentiment. Treat this as a classroom demonstration, not a production-ready model.")
    st.download_button("Download evaluation report (JSON)",data=json.dumps(metrics,indent=2),file_name="reviewwise_model_evaluation.json",mime="application/json")

elif page == "About & Limitations":
    st.subheader("What this prototype does")
    st.markdown("- Displays summary statistics and monthly rating trends.\n- Answers dataset questions using calculated statistics and relevant review retrieval.\n- Filters and exports customer review records.\n- Flags low ratings using an understandable IF–THEN rule.\n- Runs a trained text classifier on text entered by a user.\n- Displays evaluation metrics and a confusion matrix.")
    st.subheader("What this prototype does not prove")
    st.markdown("- It does not use a complete English translation of every original review.\n- The training labels are derived from ratings: 1–2 negative, 3 neutral, 4–5 positive.\n- The text model is trained only on the subset with available English text.\n- The model has not been validated by customer-service experts or deployed in a real business.\n- Estimated time savings are assumptions, not measured results.")
    st.subheader("Responsible use")
    st.write("Treat model output as a suggestion. Human staff should verify the original feedback and context before taking action. Avoid uploading private customer information to unapproved services.")
    st.subheader("Technology used")
    st.write("Python, Streamlit, pandas, scikit-learn, TF-IDF, Logistic Regression and Plotly.")

st.divider()
st.caption("ReviewWise AI • Academic demonstration • Olist customer review data • Not for production use")
