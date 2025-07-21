import streamlit as st
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Lambda
from tensorflow.keras.models import Model
import numpy as np

# ----- 1. 키워드-해시태그 매핑 엑셀 불러오기 -----
@st.cache_data
def load_keyword_hashtag_map(filepath):
    mapping_df = pd.read_excel(filepath)
    mapping_df.columns = [c.strip() for c in mapping_df.columns]
    # 실제 컬럼명 확인해서 수정 필요 (여기선 '키워드', '해시태그'로 가정)
    keyword_col = [c for c in mapping_df.columns if '키워드' in c][0]
    hashtag_col = [c for c in mapping_df.columns if '해시태그' in c][0]
    keyword_hashtag_map = {}
    for idx, row in mapping_df.iterrows():
        keyword = str(row[keyword_col]).strip()
        hashtags = [h.strip() for h in str(row[hashtag_col]).split(',') if h.strip()]
        if keyword and hashtags:
            keyword_hashtag_map[keyword] = hashtags
    return keyword_hashtag_map

# 파일 경로: 업로드한 파일 경로로 변경!
keyword_hashtag_map = load_keyword_hashtag_map("카테고리_해시태그_키워드_통합본_최종.xlsx")

# ----- 2. 해시태그 추출 함수 -----
def extract_hashtags_by_file(review, keyword_hashtag_map):
    hashtags = set()
    for keyword, tags in keyword_hashtag_map.items():
        if keyword in review:
            hashtags.update(tags)
    return list(hashtags) if hashtags else ["#리뷰추천"]

# ----- 3. 모델/토크나이저 캐싱 -----
@st.cache_resource
def load_model_and_tokenizer():
    MODEL_NAME = "klue/roberta-base"
    roberta = TFAutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    input_ids_layer = Input(shape=(256,), dtype=tf.int32, name="input_ids")
    attention_mask_layer = Input(shape=(256,), dtype=tf.int32, name="attention_mask")

    def roberta_encode(inputs):
        input_ids, attention_mask = inputs
        return roberta(input_ids=input_ids, attention_mask=attention_mask)[0]

    bert_output = Lambda(roberta_encode, output_shape=(256, 768))([input_ids_layer, attention_mask_layer])
    lstm_output = Bidirectional(LSTM(64))(bert_output)
    drop1 = Dropout(0.3)(lstm_output)
    dense1 = Dense(64, activation='relu')(drop1)
    drop2 = Dropout(0.3)(dense1)
    output = Dense(1)(drop2)

    model = Model(inputs=[input_ids_layer, attention_mask_layer], outputs=output)
    model.load_weights("roberta_bilstm_regression2.weights.h5")

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ----- 4. 평점 → 1~5 라벨 변환 -----
def map_score_to_label(score):
    if score < 1.5:
        return 1
    elif score < 2.5:
        return 2
    elif score < 3.5:
        return 3
    elif score <= 4.5:
        return 4
    else:
        return 5

# ----- 5. Streamlit UI -----
st.title("Kurly 자동평점 + 해시태그 추천 시연")
st.write("리뷰를 입력하면 로버타+BiLSTM 회귀모델이 평점과 해시태그를 예측합니다.")

review = st.text_area("리뷰를 입력하세요", "맛있고 간편해서 자주 먹게 돼요!")

if st.button("예측하기"):
    with st.spinner("예측 중..."):
        # 1. 텍스트 전처리 및 인퍼런스
        encoded = tokenizer(
            review,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='np'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        pred = model.predict([input_ids, attention_mask])[0][0]
        score = float(np.clip(pred, 1, 5))
        label = map_score_to_label(score)

        # 2. 해시태그 추출
        hashtags = extract_hashtags_by_file(review, keyword_hashtag_map)

        # 3. 결과 시각화
        st.success(f"예측 평점: **{score:.2f}점** / 라벨: **{label}점**")
        st.markdown(
            f"""<span style='font-size:36px;color:#FFC107;'>{'★'*label}{'☆'*(5-label)}</span>""",
            unsafe_allow_html=True
        )
        st.write("입력 리뷰:", review)
        st.markdown("#### 추천 해시태그")
        st.markdown(' '.join(hashtags))
