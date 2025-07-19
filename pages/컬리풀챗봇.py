import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
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

# --- 경로 설정 ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "bert_model", "tf_model.h5")
TOKENIZER_PATH = os.path.join(BASE_PATH, "bert_model")
ANSWER_CSV_PATH = os.path.join(BASE_PATH, "..", "챗봇특징추출최종.csv")

# --- 텍스트 정제 ---
def clean_text(text):
    text = re.sub(r'([a-zA-Z0-9])[^a-zA-Z0-9가-힣\s]+([a-zA-Z0-9])', r'\1 \2', str(text))
    text = re.sub(r'[^a-zA-Z0-9가-힣\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- BERT 래퍼 레이어 ---
class TFBertModelWrapper(layers.Layer):
    def __init__(self, model_name="beomi/kcbert-base", **kwargs):
        super().__init__(**kwargs)
        from transformers import TFAutoModel
        self.bert = TFAutoModel.from_pretrained(model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        return self.bert({'input_ids': input_ids, 'attention_mask': attention_mask}).last_hidden_state

# --- 모델 구조 정의 ---
def create_model():
    input_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')
    categorical_input = tf.keras.Input(shape=(CATEGORICAL_DIM,), name='categorical_input')

    bert_output = TFBertModelWrapper()(inputs=(input_ids, attention_mask))
    x = tf.keras.layers.GlobalAveragePooling1D()(bert_output)
    x = tf.keras.layers.Concatenate()([x, categorical_input])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask, categorical_input], outputs=output)
    return model

# --- 모델 및 토크나이저 로딩 ---
@st.cache_resource
def load_model_and_tokenizer():
    model = create_model()
    model.load_weights(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return model, tokenizer

# --- SBERT 로딩 ---
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

# --- 의도 예측 ---
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

# --- Streamlit 인터페이스 ---
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
