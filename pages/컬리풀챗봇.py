import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
import time
from transformers import AutoTokenizer, TFAutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import layers

# --- 커스텀 BERT 레이어 정의 ---
class TFBertModelWrapper(layers.Layer):
    def __init__(self, model_name="beomi/kcbert-base", **kwargs):
        super().__init__(**kwargs)
        self.bert = TFAutoModel.from_pretrained(model_name)
    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert({'input_ids': input_ids, 'attention_mask': attention_mask})
        return outputs.last_hidden_state

# --- 모델 구조 정의 ---
def create_model():
    input_ids = tf.keras.Input(shape=(80,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(80,), dtype=tf.int32, name="attention_mask")
    categorical_features = tf.keras.Input(shape=(64,), dtype=tf.float32, name="categorical_features")

    bert_output = TFBertModelWrapper("beomi/kcbert-base")([input_ids, attention_mask])
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(bert_output)

    x = tf.keras.layers.concatenate([pooled_output, categorical_features])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask, categorical_features], outputs=x)
    return model

# --- 설정 ---
WEIGHT_PATH = "pages/bert_model/intent_model.weights.h5"
TOKENIZER_NAME = "beomi/kcbert-base"
SBERT_MODEL = "jhgan/ko-sroberta-multitask"
CSV_FILES = {"TREND": "챗봇특징추출최종.csv"}

# --- 모델 및 리소스 로딩 함수 ---
@st.cache_resource
def load_intent_model():
    model = create_model()
    model.load_weights(WEIGHT_PATH)
    return model

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
    text = clean_text(user_input)
    X_input = tokenizer([text], padding='max_length', truncation=True, max_length=80, return_tensors='tf')
    dummy_cat = np.zeros((1, 64))
    pred = model.predict([X_input['input_ids'], X_input['attention_mask'], dummy_cat], verbose=0)
    idx2label = {0: "RECOMMEND", 1: "TREND", 2: "NEGFAQ"}
    intent_idx = np.argmax(pred, axis=1)[0]
    return idx2label.get(intent_idx, "TREND")

# --- intent별 데이터 선택 ---
def select_df_by_intent(intent, dfs):
    return dfs.get("TREND")

# --- 임베딩 사전 계산 ---
@st.cache_data
def precompute_all_embeddings(dfs, sbert_model):
    emb_dict = {}
    for key, df in dfs.items():
        if '답변' in df.columns and len(df):
            texts = df['답변'].astype(str).tolist()
            emb = sbert_model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            emb_dict[key] = {"emb": emb, "texts": texts}
    return emb_dict

# --- 유사도 기반 최적 답변 추출 ---
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

# --- Streamlit UI ---
st.set_page_config(page_title="Kurlypool 챗봇", layout="centered")
st.title("🍳 Kurlypool 챗봇")
st.markdown("리뷰 기반 간편식 추천 챗봇입니다. 아래에 질문을 입력해 주세요.")

# 리소스 로딩
intent_model = load_intent_model()
tokenizer = load_tokenizer()
sbert_model = load_sbert()
dfs = load_answer_dfs()
emb_dict = precompute_all_embeddings(dfs, sbert_model)

# 사용자 입력 처리
user_input = st.text_input("❓ 궁금한 점을 입력하세요:")

if st.button("답변 받기") and user_input.strip():
    with st.spinner("답변 생성 중..."):
        answer, intent = process_user_input(user_input, intent_model, tokenizer, dfs, emb_dict, sbert_model)
        st.markdown(f"**예측된 의도:** `{intent}`")
        st.markdown(f"**챗봇 응답:** {answer}")
