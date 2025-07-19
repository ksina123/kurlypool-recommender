import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 페이지 설정 ---
st.set_page_config(page_title="Kurlypool 챗봇", layout="centered")
st.title("🍳 Kurlypool 챗봇")
st.markdown("리뷰 기반 간편식 추천 챗봇입니다. 아래에 질문을 입력해 주세요.")

# --- 설정값 ---
MAX_LEN = 80
CATEGORICAL_DIM = 64
TOKENIZER_NAME = "beomi/kcbert-base"
SBERT_MODEL_NAME = "jhgan/ko-sroberta-multitask"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "bert_model", "tf_model.h5")
ANSWER_CSV_PATH = os.path.join(BASE_PATH, "..", "챗봇특징추출최종.csv")

# --- 텍스트 정제 함수 ---
def clean_text(text):
    text = re.sub(r'([a-zA-Z0-9])[^a-zA-Z0-9가-힣\s]+([a-zA-Z0-9])', r'\1 \2', str(text))
    text = re.sub(r'[^a-zA-Z0-9가-힣\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- BERT 래퍼 레이어 정의 ---
class TFBertModelWrapper(layers.Layer):
    def __init__(self, model_name=None, **kwargs):
        super().__init__(**kwargs)
        from transformers import TFAutoModel
        self.bert = TFAutoModel.from_pretrained(model_name or "beomi/kcbert-base")

    def call(self, inputs):
        input_ids, attention_mask = inputs
        return self.bert({'input_ids': input_ids, 'attention_mask': attention_mask}).last_hidden_state

# --- 모델 및 토크나이저 로딩 ---
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model(MODEL_PATH, custom_objects={"TFBertModelWrapper": TFBertModelWrapper})
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return model, tokenizer

# --- SBERT 모델 로딩 ---
@st.cache_resource
def load_sbert():
    return SentenceTransformer(SBERT_MODEL_NAME)

# --- 답변 데이터 로딩 ---
@st.cache_data
def load_answer_df():
    encodings = ["utf-8", "cp949", "euc-kr", "utf-8-sig"]
    for enc in encodings:
        try:
            df = pd.read_csv(ANSWER_CSV_PATH, encoding=enc)
            if "답변" in df.columns:
                df.columns = df.columns.str.strip().str.lower()
                return df
        except:
            continue
    return pd.DataFrame()

# --- 임베딩 계산 ---
@st.cache_data
def compute_embeddings(df, sbert_model):
    texts = df["답변"].astype(str).tolist()
    embeddings = sbert_model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    return texts, embeddings

# --- 의도 예측 함수 ---
def predict_intent(text, model, tokenizer):
    clean = clean_text(text)
    tokens = tokenizer([clean], padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="tf")
    dummy_cat = np.zeros((1, CATEGORICAL_DIM))
    pred = model.predict([tokens["input_ids"], tokens["attention_mask"], dummy_cat], verbose=0)
    return "RECOMMEND" if np.argmax(pred) == 0 else "TREND"

# --- 유사 답변 추출 ---
def get_best_answer(query, texts, embeddings, sbert_model):
    query_vec = sbert_model.encode([query], convert_to_tensor=False)
    sims = cosine_similarity(query_vec, embeddings)[0]
    best_idx = np.argmax(sims)
    return texts[best_idx]

# --- Streamlit UI ---
user_input = st.text_input("❓ 궁금한 점을 입력하세요:")

if st.button("답변 받기") and user_input.strip():
    with st.spinner("답변 생성 중..."):
        model, tokenizer = load_model_and_tokenizer()
        sbert_model = load_sbert()
        answer_df = load_answer_df()
        texts, emb = compute_embeddings(answer_df, sbert_model)

        intent = predict_intent(user_input, model, tokenizer)
        answer = get_best_answer(user_input, texts, emb, sbert_model)

        st.markdown(f"**예측된 의도:** `{intent}`")
        st.markdown(f"**챗봇 응답:** {answer}")
