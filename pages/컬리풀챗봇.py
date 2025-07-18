import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# --- BERT 래핑 레이어 정의 ---
class TFBertModelWrapper(layers.Layer):
    def __init__(self, model_name="beomi/kcbert-base", **kwargs):
        super(TFBertModelWrapper, self).__init__(**kwargs)
        self.bert = TFAutoModel.from_pretrained(model_name)

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert({'input_ids': input_ids, 'attention_mask': attention_mask})
        return outputs.last_hidden_state

# --- 모델 생성 함수 (2-input: BERT + CNN) ---
def create_model():
    max_len = 80
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    bert_wrapper = TFBertModelWrapper("beomi/kcbert-base")
    bert_output = bert_wrapper([input_ids, attention_mask])

    cnn_out = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')(bert_output)
    cnn_out = tf.keras.layers.GlobalMaxPooling1D()(cnn_out)

    fc = tf.keras.layers.Dense(64, activation='relu')(cnn_out)
    output = tf.keras.layers.Dense(2, activation='softmax')(fc)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    return model

# --- 경로 설정 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(CURRENT_DIR, "..", "0715_intent_model_final.h5")
TOKENIZER_NAME = "beomi/kcbert-base"
SBERT_MODEL = "jhgan/ko-sroberta-multitask"
CSV_FILES = {"TREND": "챗봇특징추출최종.csv"}

# --- 모델 및 리소스 로딩 ---
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
            for enc in encodings:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    df.columns = df.columns.str.lower().str.strip()
                    dfs[key] = df
                    break
                except:
                    continue
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
    pred = model.predict([X_input['input_ids'], X_input['attention_mask']], verbose=0)
    idx2label = {0: "RECOMMEND", 1: "TREND"}
    return idx2label.get(np.argmax(pred), "TREND")

# --- intent에 맞는 DataFrame 선택 ---
def select_df_by_intent(intent, dfs):
    return dfs.get("TREND", None)

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

# --- 유사도 기반 최적 답변 추출 ---
def get_best_answer(user_input, answer_df, emb_dict, sbert_model):
    if answer_df is None or len(answer_df) == 0:
        return "답변 후보 데이터가 없습니다."
    key = next((k for k, v in emb_dict.items() if len(v['texts']) == len(answer_df)), None)
    if key is None:
        return "임베딩 데이터가 없습니다."
    texts = emb_dict[key]['texts']
    q_embeddings = emb_dict[key]['emb']
    user_emb = sbert_model.encode([user_input], convert_to_tensor=False)
    sims = cosine_similarity(user_emb, q_embeddings)[0]
    return texts[np.argmax(sims)]

# --- 전체 파이프라인 처리 ---
def process_user_input(user_input, intent_model, tokenizer, dfs, emb_dict, sbert_model):
    intent = predict_intent(user_input, intent_model, tokenizer)
    df = select_df_by_intent(intent, dfs)
    answer = get_best_answer(user_input, df, emb_dict, sbert_model)
    return answer, intent

# --- Streamlit UI 구성 ---
st.set_page_config(page_title="Kurlypool 챗봇", layout="centered")
st.title("🍳 Kurlypool 챗봇")
st.markdown("리뷰 기반 간편식 추천 챗봇입니다. 아래에 질문을 입력해 주세요.")

try:
    intent_model = load_intent_model()
    tokenizer = load_tokenizer()
    sbert_model = load_sbert()
    dfs = load_answer_dfs()
    emb_dict = precompute_all_embeddings(dfs, sbert_model)

    user_input = st.text_input("❓ 궁금한 점을 입력하세요:")

    if st.button("답변 받기") and user_input.strip():
        with st.spinner("답변 생성 중..."):
            answer, intent = process_user_input(user_input, intent_model, tokenizer, dfs, emb_dict, sbert_model)
            st.markdown(f"**예측된 의도:** `{intent}`")
            st.markdown(f"**챗봇 응답:** {answer}")

except Exception as e:
    st.error("❗️모델 또는 리소스 로딩 중 오류 발생")
    st.exception(e)
