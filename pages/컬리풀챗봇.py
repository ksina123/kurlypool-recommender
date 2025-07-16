import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import pickle
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
    
import os
import gdown

# --- 대용량 파일(Google Drive에서 다운로드) ---
FILE_IDS = {
    "SBERT_EMB": "1vppLHbwaKbMqrJqOS8uJFUdsOgIFbh81",  # sbert_emb_cache.pkl
}
DOWNLOAD_PATHS = {
    "SBERT_EMB": "sbert_emb_cache.pkl"
}

def download_from_drive(file_id: str, save_path: str):
    if not os.path.exists(save_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, save_path, quiet=False)

# 실제 다운로드
for key in FILE_IDS:
    download_from_drive(FILE_IDS[key], DOWNLOAD_PATHS[key])

# --- 다운로드 함수 ---
def download_from_drive(file_id: str, save_path: str):
    if not os.path.exists(save_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, save_path, quiet=False)

# --- 실제 다운로드 실행 ---
for key in FILE_IDS:
    download_from_drive(FILE_IDS[key], DOWNLOAD_PATHS[key])

# --- 설정 ---
MODEL_PATH = '0715_intent_model_final.h5'  # 그대로
TOKENIZER_NAME = 'beomi/kcbert-base'
SBERT_MODEL = 'jhgan/ko-sroberta-multitask'

CSV_FILES = {
    "TREND": "챗봇특징추출최종.csv"  # 그대로 사용
}
EMB_CACHE_PATH = DOWNLOAD_PATHS["SBERT_EMB"]


# 피클 파일 사용
import pickle
with open(EMB_CACHE_PATH, "rb") as f:
    sbert_cache = pickle.load(f)

# CSV 파일 사용
import pandas as pd
df_trend = pd.read_csv(CSV_FILES["TREND"])

# --- 캐시 로딩 함수들 ---
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

# --- 임베딩 캐시 저장/로딩 ---
def save_emb_cache(emb_dict):
    with open(EMB_CACHE_PATH, "wb") as f:
        pickle.dump(emb_dict, f)

def load_emb_cache():
    if os.path.exists(EMB_CACHE_PATH):
        with open(EMB_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return None

# --- 사전 임베딩 계산 (최초 1회만)
def precompute_all_embeddings(dfs, sbert_model):
    emb_dict = load_emb_cache()
    if emb_dict:
        return emb_dict
    emb_dict = {}
    for key, df in dfs.items():
        if '답변' in df.columns and len(df):
            texts = df['답변'].astype(str).tolist()
            emb = sbert_model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
            emb_dict[key] = {"emb": emb, "texts": texts}
    save_emb_cache(emb_dict)
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

# --- 표정 이미지 intent 매핑 ---
KURLY_EMOJI = {
    "RECOMMEND": "kurly_smile.png",
    "TREND": "kurly_excited.png",
    "NEGFAQ": "kurly_sad.png",
}
def get_kurly_img_by_intent(intent):
    return KURLY_EMOJI.get(intent, "kurly_smile.png")

# --- GPT 스타일 글자 애니메이션 ---
def gpt_typing_effect(full_text, container, intent, tag_style, bubble_style, delay=0.018):
    text = ""
    for c in full_text:
        text += c
        container.markdown(
            f"""<span class="{tag_style}">{intent}</span>
                <div class="{bubble_style}">{text}</div>""",
            unsafe_allow_html=True
        )
        time.sleep(delay)
    # 마지막 완성된 버전으로 재출력 (점프 방지)
    container.markdown(
        f"""<span class="{tag_style}">{intent}</span>
            <div class="{bubble_style}">{full_text}</div>""",
        unsafe_allow_html=True
    )

# --- GPT 스타일 Streamlit UI ---
st.set_page_config(page_title="Kurlypool 챗봇", layout="centered")
st.markdown("""
<style>
body, .main, .block-container {background: #f7f7fb !important;}
.kurly-chat-wrap {margin: 0 auto; max-width: 530px;}
.user-bubble {
    background:linear-gradient(135deg,#f4f0ff,#e2e5fc);
    color:#2c1955; font-size:1.13rem;
    border-radius:17px 16px 8px 19px;
    padding:13px 15px 12px 17px; margin:9px 0 0 43px; text-align:right;
    min-width:72px; max-width:92%; float:right; clear:both;
    box-shadow:0 2px 8px #ece3fa5a;
    border:1px solid #ece6fc;
}
.kurly-bubble {
    background:linear-gradient(135deg,#fff,#f6f3ff);
    border:1.2px solid #eadcff;
    color:#453174; font-size:1.13rem;
    border-radius:16px 19px 19px 8px;
    padding:14px 18px 12px 14px;
    min-width:90px; max-width:88%;
    margin:0; display:inline-block;
    box-shadow:0 2px 10px #f2edff7e;
}
.intent-tag {
    display:inline-block; background:#e7ddff; color:#7758c1;
    font-size:1.00rem; border-radius:9px;
    padding:3px 13px 2px 13px; margin-bottom:4px;
    font-weight:600; letter-spacing:0.01em;
}
.kurly-title {
    font-size:2.15rem;
    font-weight:900;
    color:#704ed9;
    margin-bottom:0.28em;
    letter-spacing:-0.5px;
}
.kurly-desc {
    color:#968bb8;
    font-size:1.08rem;
    margin-bottom:18px;
    margin-top:-5px;
}
.stTextInput>div>div>input {
    background: #f7f4ff;
    border: 1.2px solid #e2dcff;
    border-radius: 11px;
    padding: 10px 13px;
    font-size: 1.07rem;
    color: #39266e;
    box-shadow: 0 1.5px 8px #eae6ff7c;
}
.stTextInput>div>div>input:focus {
    outline: none;
    border: 1.5px solid #9b81e7;
}
.stButton>button {
    background: linear-gradient(90deg,#a68cf8 0%,#8055e7 100%);
    color: #fff;
    font-weight: 700;
    border-radius: 11px;
    border: none;
    font-size: 1.13rem;
    height: 46px;
    margin-top:8px;
    box-shadow: 0 2px 10px #dcd0ff60;
    transition: 0.13s;
}
.stButton>button:hover {
    background: linear-gradient(90deg,#c3b6ed 0%,#a48cf4 100%);
}
</style>
<div class='kurly-chat-wrap'></div>
""", unsafe_allow_html=True)

# --- 상단 Kurly 캐릭터 + 타이틀 ---
logo_cols = st.columns([1, 7])
with logo_cols[0]:
    st.image("kurly_bot.png", width=64)
with logo_cols[1]:
    st.markdown(
        "<span class='kurly-title'>Kurlypool 챗봇</span>",
        unsafe_allow_html=True
    )
st.markdown("<div class='kurly-desc'>궁금한 점을 자유롭게 물어보세요!</div>", unsafe_allow_html=True)

# 세션 대화 내역 및 애니메이션 플래그
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "animate_last" not in st.session_state:
    st.session_state.animate_last = False

# 모델 및 캐시 세션 (기존과 동일)
if 'intent_model' not in st.session_state:
    with st.spinner("모델 로딩 중..."):
        st.session_state.intent_model = load_intent_model()
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = load_tokenizer()
if 'dfs' not in st.session_state:
    st.session_state.dfs = load_answer_dfs()
if 'sbert_model' not in st.session_state:
    st.session_state.sbert_model = load_sbert()
if 'emb_dict' not in st.session_state:
    with st.spinner("임베딩 사전 계산..."):
        st.session_state.emb_dict = precompute_all_embeddings(
            st.session_state.dfs, st.session_state.sbert_model
        )

# --- 대화 내역 출력 (유저/챗봇 분리, 표정 이미지 st.image로 + 마지막 챗봇 메시지 GPT 애니) ---
for i, msg in enumerate(st.session_state.chat_history):
    if msg['role'] == "user":
        st.markdown(f"""<div class="user-bubble">{msg['content']}</div>""", unsafe_allow_html=True)
    else:
        img_path = get_kurly_img_by_intent(msg['intent'])
        cols = st.columns([1, 11])
        with cols[0]:
            st.image(img_path, width=48)
        with cols[1]:
            # 마지막 챗봇 메시지 & animate_last=True 일 때만 애니메이션
            if (
                i == len(st.session_state.chat_history) - 1
                and st.session_state.get("animate_last", False)
            ):
                effect_box = st.empty()
                gpt_typing_effect(
                    msg['content'],
                    effect_box,
                    msg['intent'],
                    "intent-tag",
                    "kurly-bubble",
                    delay=0.016
                )
                st.session_state.animate_last = False
            else:
                st.markdown(
                    f"""<span class="intent-tag">{msg['intent']}</span>
                        <div class="kurly-bubble">{msg['content']}</div>""",
                    unsafe_allow_html=True
                )

# --- 입력 폼 (GPT 스타일) ---
with st.form(key="user_input_form", clear_on_submit=True):
    user_input = st.text_input("", placeholder="여기에 질문을 입력하세요!", key="input_text")
    submitted = st.form_submit_button("전송", use_container_width=True)

if submitted and user_input.strip():
    answer, intent = process_user_input(
        user_input,
        st.session_state.intent_model,
        st.session_state.tokenizer,
        st.session_state.dfs,
        st.session_state.emb_dict,
        st.session_state.sbert_model
    )
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "intent": intent})
    st.session_state.animate_last = True
    st.rerun()
