import streamlit as st
import pandas as pd
import urllib.parse
import os
import gdown

# --- Google Drive에서 파일 다운로드 함수 ---
@st.cache_data
def download_from_google_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

# --- Google Drive 파일 ID 및 경로 ---
drive_file_id = "10vpECluTJxphgAhIAkBREOMcp-ZnjU_e"
csv_path = "데이터.csv"

# --- 파일 다운로드 ---
download_from_google_drive(drive_file_id, csv_path)

# --- 데이터 로드 ---
df = pd.read_csv(csv_path)
cat_df = df.copy()  # 중복 로드 대신 복사
avg_df = pd.read_csv("평균예측평점최종.csv")  # ← 이 줄이 반드시 먼저!
category_col = "카테고리"                    # ← 컬럼명 실제 데이터에 맞게 수정
categories = cat_df[category_col].dropna().unique().tolist()
categories = ["모든 카테고리"] + sorted(categories)

def clean_text(x):
    if isinstance(x, str):
        return x.strip().replace('\n', '').replace('\r', '')
    return x

df["상품명"] = df["상품명"].apply(clean_text)
avg_df["상품명"] = avg_df["상품명"].apply(clean_text)
score_col = next((c for c in avg_df.columns if "평균예측평점" in c or "예측평점" in c or "score" in c.lower()), None)
if score_col is None:
    st.error("❌ 평점 관련 열을 찾을 수 없습니다.")
    st.stop()

st.set_page_config(page_title="Kurly Meal Solution 데이터보드", layout="wide", page_icon="🛒")

# --- 스타일(로고, 검색창) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, .stApp {
    background: #fff !important;
    color: #222 !important;
    font-family: 'Montserrat', 'Spoqa Han Sans Neo', sans-serif !important;
}
.kurly-header {
    display: flex; align-items: center; justify-content: flex-start; gap: 32px; margin-top: 18px; margin-left: 34px;
}
.kurly-logo-all {
    font-family: 'Pacifico', cursive !important;
    font-size: 2.6rem;
    color: #800080;
    font-weight: 700;
    text-shadow: 0 2px 8px #f4e4ff70;
    letter-spacing: 1px;
}
div[data-testid="stTextInput"] input {
    border: 2.2px solid #a275e3; border-radius: 11px;
    padding: 0.72em 1.25em;
    font-size: 1.14rem; width: 320px;
    outline: none; box-shadow: 0 2px 13px #ede3fc32;
    background: #fff;
    color: #3d0072;
    font-family: 'Montserrat', 'Spoqa Han Sans Neo', sans-serif;
    transition: border 0.22s;
}
div[data-testid="stTextInput"] input:focus {
    border: 2.7px solid #8600e6;
    background: #f6f3fd;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #55516b;
    font-weight: 500;
    font-size: 1.07rem;
    opacity: 0.93;
    letter-spacing: -0.2px;
}
.kurly-hashtagbar {
    margin-top: 8px; margin-bottom: 0;
    display: flex; flex-wrap: wrap; gap: 7px 8px;
}
.kurly-hashtag {
    background: #f5f1ff; color: #9100ce; font-weight: 600;
    font-size: 0.96rem; border-radius: 8px;
    padding: 4px 15px 4px 12px; margin-right: 0;
    box-shadow: 0 1.5px 7px #e8d9f5c7;
    border: 1.2px solid #e1d0f9;
    display: inline-block; transition: background 0.18s;
}
.kurly-hashtag:hover {
    background: #ede5fa; color: #8600e6; font-weight: 700;
    box-shadow: 0 2.5px 13px #cbb3f7;
}
.kurly-pagebar {
    display: flex; justify-content: center; gap: 7px; margin: 30px 0 36px 0;
    background: #fff; color: #9100ce !important; font-weight: bold;
    font-size: 1.08rem; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2.5px 9px #eee9fa;
}
.kurly-btn {
    width: 39px; height: 39px; border-radius: 50%; border: 2px solid #e5d7f8;
    background: #fff; color: #8600e6 !important; font-weight: bold; font-size: 1.07rem;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2.5px 9px #eee9fa;
    cursor: pointer; transition: .17s; text-decoration: none !important; outline: none !important; border-bottom: none !important;
}
.kurly-btn.selected, .kurly-btn.selected:hover {
    background: #8600e6 !important; color: #fff !important; border: 2px solid #8600e6 !important;
    box-shadow: 0 2.5px 13px #cbb3f7 !important;
}
.kurly-btn:hover, .kurly-btn:active, .kurly-btn:visited, .kurly-btn:focus {
    background: #f1e3fa !important;
    color: #8600e6 !important;
    text-decoration: none !important;
    outline: none !important;
    border-bottom: none !important;
}
</style>
<div class="kurly-header">
  <span class="kurly-logo-all">Kurly Meal Solution</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr style="border:0;border-top:1.5px solid #eee;margin-top:10px;margin-bottom:4px;">', unsafe_allow_html=True)
st.markdown("""
<style>
div[data-baseweb="select"] > div {
    background-color: #fff !important;
    color: #8000c2 !important;
    border-radius: 8px !important;
    border: 2px solid #c4a7e7 !important;
}
div[data-baseweb="select"] span {
    color: #8000c2 !important;
    font-weight: 600 !important;
}
div[data-baseweb="popover"] {
    background-color: #fff !important;
}
div[data-baseweb="optio {
    background-color: #fff !important;
    color: #8000c2 !important;
    font-weight: 600 !important;
}
div[data-baseweb="option"]:hover {
    background-color: #f3e7fd !important;n"]
    color: #8600e6 !important;
}
</style>
""", unsafe_allow_html=True)

# --- 별점 표시 함수 ---
def render_star_html(score: float) -> str:
    style = """
    <style>
    .star {
        font-size: 20px;
        display: inline-block;
        width: 1em;
        overflow: hidden;
        margin-right: 2px;
        vertical-align: middle;
        position: relative;
    }
    </style>
    """
    html = style
    for i in range(5):
        left = min(max(score - i, 0), 1)
        percent = int(left * 100)
        if percent == 0:
            html += '<span class="star" style="color:#e0e0e0;">★</span>'
        elif percent == 100:
            html += '<span class="star" style="color:#FFC107;">★</span>'
        else:
            html += f'''
            <span class="star" style="
                width:1em; overflow:hidden; display:inline-block;
                background: linear-gradient(90deg, #FFC107 {percent}%, #e0e0e0 {percent}%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-fill-color: transparent;
                color: #FFC107;
            ">★</span>
            '''
    return html


# --- 해시태그 추출 ---
def extract_top_hashtags(product, topn=3, min_reviews=5):
    product_reviews = df[df["상품명"] == product].copy()
    if product_reviews.shape[0] == 0:
        return []
    high_score_reviews = product_reviews[product_reviews["예측평점"] >= 4.0]
    if len(high_score_reviews) < min_reviews:
        filtered = product_reviews
    else:
        high_score_reviews.loc[:, "리뷰길이"] = high_score_reviews["리뷰"].astype(str).str.len()
        length_threshold = high_score_reviews["리뷰길이"].quantile(0.8)
        filtered = high_score_reviews[high_score_reviews["리뷰길이"] >= length_threshold]
        if filtered.empty:
            filtered = high_score_reviews
    if filtered.shape[0] == 0:
        return []
    from collections import Counter
    all_tags = []
    for h in filtered["해시태그"].dropna():
        if not str(h).strip():
            continue
        tags = [t.strip() for t in h.replace("#", " #").split() if t.startswith("#")]
        for tag in tags:
            text = tag.replace("#", "").strip()
            if any(ex in text for ex in ["에어프라이어", "아이", "신랑"]):
                continue
            text = "".join([c for c in text if ("\uAC00" <= c <= "\uD7A3") or c.isalnum()])
            if text and not text.isspace():
                all_tags.append(text)
    tag_counts = Counter(all_tags)
    result = [tag for tag, _ in tag_counts.most_common(topn)]
    return result

# --- 상세 페이지 ---
import re
from collections import Counter
import pandas as pd

def clean_tag(tag):
    return re.sub(r"^[^\w가-힣]+|[^\w가-힣]+$", "", str(tag).strip())

def show_detail(product):
    st.header(f"{product} 리뷰 상세")
    match = df["상품명"].apply(lambda x: x == product)
    data = df[match]
    if data.empty:
        st.error("해당 상품이 없습니다.")
        st.markdown('<a href="/" style="font-size:1.1rem; color:#8600e6;">← 메인으로</a>', unsafe_allow_html=True)
        return

    vals = avg_df[avg_df["상품명"] == product][score_col].values
    avg_score = float(vals[0]) if vals.size else 0.0

    cols = st.columns([1, 2])  # [왼쪽: 이미지][오른쪽: 키워드게이지]

    with cols[0]:
        imgs = data["이미지URL"].dropna().tolist()
        if imgs:
            st.image(imgs[0], width=320)
        else:
            st.write("이미지 없음")
        st.markdown(render_star_html(avg_score) + f" **({avg_score:.1f}점)**", unsafe_allow_html=True)

    with cols[1]:
        # 키워드 집계 및 게이지 표시
        all_keywords = []
        for kws in data["예측키워드"].dropna():
            for t in re.split(r"[,/]", str(kws)):
                tag = clean_tag(t)
                if tag:
                    all_keywords.append(tag)
        keyword_counts = Counter(all_keywords)
        total_reviews = data.shape[0] if data.shape[0] else 1
        top_keywords = keyword_counts.most_common(5)

        # 키워드 게이지 카드(오른쪽 컬럼, 가로 최대)
        if top_keywords:
            card_html = '''
            <div style="width:100%;display:flex;flex-direction:column;gap:14px 0;margin:14px 0 10px 0;">
            '''
            colors = ["#2580e7", "#895ff5", "#38b000", "#fd9100", "#ff4365"]
            for i, (tag, cnt) in enumerate(top_keywords):
                color = colors[i % len(colors)]
                percent = cnt / total_reviews * 100
                crown = (
                    '<img src="https://cdn-icons-png.flaticon.com/128/1828/1828884.png" '
                    'width="26" style="margin-right:7px;vertical-align:-7px;filter:drop-shadow(0 0 5px gold) drop-shadow(0 0 4px #e7be3a);">'
                    if i == 0 else ""
                )
                chip_html = (
                    f'<div style="background:{color}15;padding:14px 19px 17px 19px;border-radius:19px;display:flex;flex-direction:column;justify-content:center;min-width:210px;max-width:99%;">'
                    f'<div style="display:flex;align-items:center;">'
                    f'{crown}<span style="color:{color};font-size:1.13rem;font-weight:700;">{tag}</span>'
                    f'<span style="margin-left:14px;background:{color};color:#fff;border-radius:11px;padding:2px 13px 2px 13px;font-size:1rem;font-weight:500;">{cnt}회</span>'
                    f'<span style="margin-left:10px;color:#555;font-size:0.99rem;">({percent:.1f}%)</span>'
                    f'</div>'
                    f'<div style="background:#e7f0fa;width:100%;height:13px;border-radius:7px;margin-top:8px;position:relative;">'
                    f'  <div style="background:{color};width:{max(percent,6):.0f}%;height:100%;border-radius:7px;"></div>'
                    f'</div>'
                    f'</div>'
                )
                card_html += chip_html
            card_html += '</div>'
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("---")

    # 리뷰 리스트 출력
    if data["리뷰"].dropna().shape[0] == 0:
        st.markdown(
            '<div style="color:#a035ca;font-weight:600;font-size:1.09rem; background:#f9f3ff; border-radius:10px; padding:13px 18px; display:inline-block;">아직 등록된 리뷰가 없습니다.<br>가장 먼저 리뷰를 남겨보세요!</div>',
            unsafe_allow_html=True
        )
    else:
        for _, row in data.iterrows():
            if pd.isna(row.get("리뷰")) or not str(row.get("리뷰")).strip():
                continue
            st.write(row["리뷰"])
            if pd.notna(row.get("예측평점")):
                r = float(row["예측평점"])
                st.markdown(render_star_html(r) + f" **({int(r)}점)**", unsafe_allow_html=True)
            if pd.notna(row.get("해시태그")):
                st.write(row["해시태그"])
            st.markdown("---")
    st.markdown('<a href="/" style="font-size:1.1rem; color:#8600e6;">← 메인으로</a>', unsafe_allow_html=True)

# --- 메인 ---
def show_main():
    cols = st.columns([2, 10])  # 왼쪽: 검색, 오른쪽: 빈공간
    with cols[0]:
        search_term = st.text_input(
    label="검색어 입력",
    placeholder="검색어를 입력해 주세요",
    key="main_search",
    label_visibility="collapsed"
)
    with cols[1]:
        st.write("")
    
    # 2. 카테고리 선택
    selected_category = st.selectbox("카테고리를 선택하세요", categories, index=0)
    
    # 3. 상품 통계 데이터 생성
    df_prod_stat = (
        df.groupby("상품명")
        .agg(리뷰수=('리뷰', 'count'))
        .reset_index()
        .merge(avg_df[["상품명", score_col]], on="상품명", how="left")
    )
    
    # 4. 검색어 필터 적용
    if search_term:
        mask = df_prod_stat["상품명"].fillna("").str.lower().str.contains(search_term.lower())
        filtered = df_prod_stat[mask].copy()
    else:
        filtered = df_prod_stat.copy()
    
    # 5. 카테고리 필터 적용
    if selected_category != "모든 카테고리":
        prod_names = cat_df[cat_df[category_col] == selected_category]["상품명"].tolist()
        filtered = filtered[filtered["상품명"].isin(prod_names)]
    
    if filtered.empty:
        st.warning(f"'{search_term}'(으)로 검색된 상품이 없습니다.")
        return
    
    # 6. 정렬 옵션
    sort_option = st.selectbox(
        "정렬 기준을 선택하세요",
        ("리뷰 많은 순", "리뷰 적은 순", "평점 높은 순", "평점 낮은 순", "상품명 가나다순"),
        index=0
    )
    if sort_option == "리뷰 많은 순":
        filtered = filtered.sort_values("리뷰수", ascending=False)
    elif sort_option == "리뷰 적은 순":
        filtered = filtered.sort_values("리뷰수", ascending=True)
    elif sort_option == "평점 높은 순":
        filtered = filtered.sort_values(score_col, ascending=False)
    elif sort_option == "평점 낮은 순":
        filtered = filtered.sort_values(score_col, ascending=True)
    elif sort_option == "상품명 가나다순":
        filtered = filtered.sort_values("상품명", ascending=True)
    
    # 7. 카드/페이지네이션 및 출력
    prods = filtered["상품명"].tolist()
    PAGE_SIZE = 8
    page = int(st.query_params.get("page", [1])[0])
    total = len(prods)
    total_pages = (total - 1) // PAGE_SIZE + 1
    group_size = 10
    page_group = (page - 1) // group_size
    start_p = page_group * group_size + 1
    end_p = min(start_p + group_size - 1, total_pages)
    page_range = list(range(start_p, end_p + 1))

    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_prods = prods[start:end]
    cols = st.columns(4)
    for idx, p in enumerate(page_prods):
        with cols[idx % 4]:
            imgs = df[df["상품명"] == p]["이미지URL"].dropna().tolist()
            preview = df[df["상품명"] == p]["리뷰"].dropna().astype(str).tolist()
            vals = avg_df[avg_df["상품명"] == p][score_col].values
            sc = round(float(vals[0]), 1) if vals.size else 0
            safe_p = urllib.parse.quote_plus(p)
            top_tags = extract_top_hashtags(p, 3)
            review_cnt = df[df["상품명"] == p]["리뷰"].dropna().shape[0]
            hashtag_html = ""
            if top_tags:
                hashtag_html = "<div class='kurly-hashtagbar'>" + "".join(
                    f"<span class='kurly-hashtag'>{tag}</span>" for tag in top_tags
                ) + "</div>"
            img_html = (
                f"<img src='{imgs[0]}' style='width:100%; max-width:210px; height:210px; border-radius:9px; margin-bottom:10px; display:block; object-fit:cover; object-position:center;' />"
                if imgs else
                "<div style='width:100%; max-width:210px; height:210px; background:#f4f4f4; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#bbb; margin-bottom:10px;'>이미지 없음</div>"
            )
            st.markdown(
                f"<a href='/?product={safe_p}' style='text-decoration:none;color:inherit;'>"
                f"<div style='padding:22px 16px 16px 16px; background:#fff; border-radius:14px; box-shadow:0 2px 14px #2222; cursor:pointer; transition:0.2s; margin-bottom:28px; margin-right:20px; border:1.5px solid #eee; min-height:320px;'>"
                f"<div style='font-weight:700;font-size:1.18rem;line-height:1.5;margin-bottom:10px; color:#1a1a1a;'>{p}</div>"
                f"{img_html}"
                f"<div style='color:#444;font-size:1.03rem;margin-bottom:10px;'>{preview[0][:40]+'...' if preview else ''}</div>"
                f"<div style='margin-bottom:6px; display:flex; align-items:center;'>"
                f"{render_star_html(sc) if vals.size else ''}"
                f"<span style='color:#555;font-size:0.98rem; margin-left:4px;'>{f'({sc:.1f}점)' if vals.size else ''}</span>"
                f"<span style='color:#8e24aa;font-size:0.97rem; margin-left:8px;'>리뷰수 {review_cnt}</span>"
                f"</div>"
                f"{hashtag_html}"
                f"</div></a>",
                unsafe_allow_html=True
            )

    nav_html = '<div class="kurly-pagebar">'
    if start_p > 1:
        nav_html += f'<a class="kurly-btn" href="?page={start_p-1}">〈</a>'
    for n in page_range:
        if n == page:
            nav_html += f'<a class="kurly-btn selected" href="?page={n}">{n}</a>'
        else:
            nav_html += f'<a class="kurly-btn" href="?page={n}">{n}</a>'
    if end_p < total_pages:
        nav_html += f'<a class="kurly-btn" href="?page={end_p+1}">〉</a>'
    nav_html += '</div>'
    st.markdown(nav_html, unsafe_allow_html=True)

# --- 라우팅 ---
params = st.query_params
product = params.get("product", None)
if isinstance(product, list): product = product[0]
if product:
    product = str(product)
    show_detail(product)
else:
    show_main()
    
# --- 챗봇 아이콘 (오른쪽 하단 고정) ---
import base64

def load_image_base64(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode()

# base64 변환
img_base64 = load_image_base64("kurly_bot.png")

# 삽입
st.markdown(f"""
<style>
.kurly-chatbot-button {{
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 9999;
}}
.kurly-chatbot-button img {{
    width: 80px;
    cursor: pointer;
    transition: transform 0.2s;
    filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.25));
}}
.kurly-chatbot-button img:hover {{
    transform: scale(1.07);
}}
</style>
<div class="kurly-chatbot-button">
  <a href="https://kurlypool-recommender.streamlit.app/컬리풀챗봇" target="_blank">
    <img src="data:image/png;base64,{img_base64}" alt="챗봇">
  </a>
</div>
""", unsafe_allow_html=True)

