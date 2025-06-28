import streamlit as st

# 페이지 메타 설정
st.set_page_config(
    page_title="D3 Lens",
    page_icon="❄️",
    layout="wide",
)

def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-image:
                linear-gradient(rgba(30,40,60,0.7), rgba(30,40,60,0.7)),
                url('https://i.imgur.com/RLtYcWs.gif');
            background-blend-mode: multiply;
            background-attachment: fixed;
            background-size: cover;
        }
        /* 중앙 컨텐츠 정렬 및 여백 */
        .main > div {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding-top: 2vh;
            min-height: auto;
        }
        /* 카드 스타일 */
        .feature-box {
            background: rgba(255,255,255,0.92);
            border-radius: 1.2rem;
            padding: 2.2rem 2.5rem;
            box-shadow: 0 6px 24px rgba(0,0,0,0.10);
            font-size: 1.13em;
            line-height: 1.7;
            margin-top: 2.5rem;
            max-width: 600px;
        }
        /* 제목 스타일 */
        .main-title {
            font-family: 'Pretendard', 'Nanum Gothic', 'Malgun Gothic', Arial, sans-serif;
            font-size: 3.6em;
            font-weight: 700;
            color: #fff !important;
            letter-spacing: -1px;
            margin-bottom: 0.3em;
            text-shadow: 0 2px 12px rgba(30,40,60,0.25);
        }
        .subtitle {
            font-size: 1.3em;
            color: #fff !important;
            margin-bottom: 1.8em;
            text-shadow: 0 1px 8px rgba(30,40,60,0.18);
        }
        .accent {
            color: #2E8BC0;
            font-weight: 600;
        }
        /* 사이드바 색상 */
        [data-testid="stSidebar"] {
            background-color: #E8F0F8;
        }
        /* 헤더 색상 */
        [data-testid="stHeader"]{
            background-color: rgba(0,0,0,0);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()


# 본문 구성
st.markdown("""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
    <h1 class="main-title">D3 Lens</h1>
    <h1 class="subtitle">Snowflake 데이터 기반 백화점 커버리지 분석</h1>
    <div class="feature-box" style="margin-left:auto; margin-right:auto;">
        <ul style="list-style:square inside; padding-left:0;">
            <li>지역별 백화점 <span class="accent">커버리지 레벨</span> 히트맵</li>
            <li><span class="accent">소비 / 충성도 / 방문율</span> KPI 시각화</li>
            <li>지역별 주민 <span class="accent">소비 라이프스타일 세그먼트</span> 분석</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)