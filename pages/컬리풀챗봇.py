import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
import time
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 커스텀 BERT 레이어 정의 ---
from tensorflow.keras import layers
class TFBertModelWrapper(layers.Layer):
    def __init__(self, model_name="beomi/kcbert-base", **kwargs):
        super().__init__(**kwargs)
        self.bert = TFAutoModel.from_pretrained(model_name)
    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert({'input_ids': input_ids, 'attention_mask': attention_mask})
        return outputs.last_hidden_state

# --- 설정 ---
MODEL_PATH = '0715_intent_model_final.h5'
TOKENIZER_NAME = 'beomi/kcbert-base'
SBERT_MODEL = 'jhgan/ko-sroberta-multitask'
CSV_FILES = {
    "TREND": "챗봇특징추출최종.csv"
}

@st.cache_resource
def load_intent_model():
    return load_model(MODEL_PATH, custom_objects={'TFBertModelWrapper': TFBertModelWrapper})

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)

@st.cache_resource
def load_sbert():
    return SentenceTransformer(SBERT_MODEL)

@st.cache_data
def load_answer_dfs():
    dfs = {}
    for key, path in CSV_FILES.items():
        if os.path.exists(path):
            encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except:
                    continue
            if df is not None:
                df.columns = df.columns.str.lower().str.strip()
                dfs[key] = df
    return dfs

# --- 텍스트 정제 ---
def clean_text(text):
    text = re.sub(r'([a-zA-Z0-9])[^a-zA-Z0-9가-힣\s]+([a-zA-Z0-9])', r'\1 \2', str(text))
    text = re.sub(r'[^a-zA-Z0-9가-힣\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 의도 예측 ---
def predict_intent(user_input, model, tokenizer):
    try:
        text = clean_text(user_input)
        X_input = tokenizer([text], padding='max_length', truncation=True, max_length=80, return_tensors='tf')
        dummy_cat = np.zeros((1, 64))
        pred = model.predict([X_input['input_ids'], X_input['attention_mask'], dummy_cat], verbose=0)
        idx2label = {0: "RECOMMEND", 1: "TREND", 2: "NEGFAQ"}
        intent_idx = np.argmax(pred, axis=1)[0]
        return idx2label.get(intent_idx, "TREND")
    except Exception:
        return "TREND"

# --- intent별 데이터 선택 ---
def select_df_by_intent(intent, dfs):
    return dfs.get("TREND")

# --- 사전 임베딩 계산 ---
@st.cache_data
def precompute_all_embeddings(dfs, sbert_model):
    emb_dict = {}
    for key, df in dfs.items():
        if '답변' in df.columns and len(df):
            texts = df['답변'].astype(str).tolist()
            emb = sbert_model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            emb_dict[key] = {"emb": emb, "texts": texts}
    return emb_dict

# --- 베스트 답변 유사도 계산 ---
def get_best_answer(user_input, answer_df, emb_dict, sbert_model):
    if answer_df is None or len(answer_df) == 0:
        return "답변 후보 데이터가 없습니다."
    key = None
    for k, v in emb_dict.items():
        if answer_df is not None and 'texts' in v and len(v['texts']) == len(answer_df):
            key = k
            break
    if key is None:
        return "임베딩 데이터가 없습니다."
    texts = emb_dict[key]['texts']
    q_embeddings = emb_dict[key]['emb']
    user_emb = sbert_model.encode([user_input], convert_to_tensor=False)
    sims = cosine_similarity(user_emb, q_embeddings)[0]
    best_idx = sims.argmax()
    return texts[best_idx]

# --- 전체 파이프라인 처리 ---
def process_user_input(user_input, intent_model, tokenizer, dfs, emb_dict, sbert_model):
    intent = predict_intent(user_input, intent_model, tokenizer)
    df = select_df_by_intent(intent, dfs)
    answer = get_best_answer(user_input, df, emb_dict, sbert_model)
    return answer, intent

# --- UI 테스트용 (예시) ---
if __name__ == "__main__":
    st.set_page_config(page_title="Kurlypool 챗봇", layout="centered")
    st.title("💬 Kurlypool 챗봇")

    model = load_intent_model()
    tokenizer = load_tokenizer()
    sbert_model = load_sbert()
    dfs = load_answer_dfs()
    emb_dict = precompute_all_embeddings(dfs, sbert_model)

    user_input = st.text_input("검색", placeholder="무엇을 도와드릴까요?", key="main_search", label_visibility="collapsed")

    if st.button("검색"):
        if user_input:
            answer, intent = process_user_input(user_input, model, tokenizer, dfs, emb_dict, sbert_model)
            st.markdown(f"**의도:** `{intent}`")
            st.markdown(f"**답변:** {answer}")
