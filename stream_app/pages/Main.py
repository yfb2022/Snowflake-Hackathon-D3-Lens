import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import json
import pydeck as pdk
from sklearn.preprocessing import MinMaxScaler
from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import plotly.graph_objects as go
from dotenv import load_dotenv
import plotly.express as px
from millify import millify
import yaml
import json
import gdown
import os


load_dotenv()
api_key = os.getenv('OPEN_API_KEY')
with open("./assets/marketing_prompt.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 페이지 설정
st.set_page_config(layout="wide")

# ---------------------------
# 샘플 데이터
# ---------------------------
@st.cache_data
def load_data():
    with open("assets/data_id.json", "r") as f:
        data_files = json.load(f)

    read_options = {
        "rci": {"engine": "python"},
        "visit_rate": {"engine": "python"}
    }
    data = {}
    for name, file_path in data_files.items():
        options = read_options.get(name, {})
        url = f"https://drive.google.com/uc?id={file_path}"
        output = os.path.join("./assets", f"{name}.csv")
        gdown.download(url, output, quiet=True)
        data[name] = pd.read_csv(output, **options)

    return (
        data["customer_info"],
        data["metric"],
        data["rci"],
        data["vip_rate"],
        data["visit_rate"],
        data["metric_zscored"]
    )

customer_info, metric, rci, vip_rate, visit_rate, metric_zscored = load_data()
logo_image = Image.open('./assets/logo.png')
# ---------------------------
# 사이드바 필터
# ---------------------------

st.sidebar.header("Filters")
dept = st.sidebar.selectbox("Departmentstore", sorted(list(metric['DEP_NAME'].unique())))
region = st.sidebar.selectbox("Region", sorted(list(metric['ADDR_LV2'][metric['DEP_NAME'] == dept].unique())))
dong = st.sidebar.selectbox("Dong", sorted(list(metric['ADDR_LV3'][(metric['DEP_NAME'] == dept) & (metric['ADDR_LV2'] == region)].unique())))

metric['STANDARD_YEAR_MONTH'] = pd.to_datetime(metric['STANDARD_YEAR_MONTH']) 
month_list = sorted(metric['STANDARD_YEAR_MONTH'].dt.strftime("%Y-%m").unique())
period = st.sidebar.select_slider(
    "Period",
    options=month_list,
    value=(month_list[0], month_list[-1])
)

st.sidebar.image(logo_image, use_container_width=True)

filtered = metric[
    (metric['DEP_NAME'] == dept) &
    (metric['ADDR_LV2'] == region) &
    (metric['ADDR_LV3'] == dong) &
    (metric['STANDARD_YEAR_MONTH'] >= period[0]) &
    (metric['STANDARD_YEAR_MONTH'] <= period[1])
]

compared = metric[
    (metric['DEP_NAME'] == dept) &
    (metric['ADDR_LV2'] == region) &
    (metric['ADDR_LV3'] == dong) &
    (metric['STANDARD_YEAR_MONTH'] <= period[0])
]

zscored = metric_zscored[
    (metric_zscored['CITY_KOR_NAME'] == region) &
    (metric_zscored['DISTRICT_KOR_NAME'] == dong)]

# ---------------------------
# KPI 지표 상단
# ---------------------------

st.title("🏬 동네별 백화점 커버리지 한눈에 보기")
st.caption("Last updated: May 14, 2025")
st.markdown(" ")

col1, col2, col3 = st.columns(3) # Glass Morph가 필요하다.
def mean_col(col, df):
    return float(df[col].mean())

def changer(df1, df2, unit):
    change = round(df1 - df2, 2)
    if unit == '₩':
        change_won = millify(round(df1 - df2, 2), precision=1)
        formatted_change = f"{change_won}"
    else:
        formatted_change = f"{change}{unit}"
    return formatted_change

col1.metric("VIP 방문률",
            f'{round(mean_col('VIP_RATE', filtered) * 100, 2)}%',
            #'updated',
            border = True
)

col2.metric("커버리지 지수",
            f'{round(mean_col('COVERAGE_SCORE', filtered), 2)} / 1.0',
            changer(mean_col('COVERAGE_SCORE', filtered) , mean_col('COVERAGE_SCORE', compared), 'pt'),
            border = True
)

col3.metric("평균 추정 매출",
            millify(round(mean_col('SALES_ADDR', filtered), 2), precision= 1),
            changer(mean_col('SALES_ADDR', filtered) , mean_col('SALES_ADDR', compared), '₩'),
            border = True
            )

# ---------------------------
# 지도 Placeholder (추후 pydeck 대체 가능)
# ---------------------------
def flatten_polygon(polygon):
    while isinstance(polygon, list) and isinstance(polygon[0], list):
        if isinstance(polygon[0][0], (float, int)):
            break
        polygon = polygon[0]
    return polygon

def polygon_centroid(polygon):
    coords = np.array(flatten_polygon(polygon))
    lon_mean = coords[:, 0].mean()
    lat_mean = coords[:, 1].mean()
    return [lon_mean, lat_mean]

def df_filter(df, store, period, score):
    dept = store
    filtered = df[
        (df['DEP_NAME'] == dept) &
        (df['STANDARD_YEAR_MONTH'] >= period[0]) &
        (df['STANDARD_YEAR_MONTH'] <= period[1])
    ].groupby(['DEP_NAME', 'ADDR_LV2', 'ADDR_LV3', 'DISTRICT_GEOM'])[score].mean().reset_index()
    filtered['DISTRICT_GEOM'] = filtered['DISTRICT_GEOM'].apply(json.loads).apply(lambda x: x['coordinates'])
    filtered['COORD'] = filtered['DISTRICT_GEOM'].apply(lambda x : x[0])
    filtered[score] = filtered[score].apply(lambda x : round(x , 2))
    return filtered

def layer_vis(df, score, type):
    scaler = MinMaxScaler()
    df['SCORE'] = scaler.fit_transform(np.log1p(df[[score]]))
    if type == 'Poly':
        layer = pdk.Layer(
            'PolygonLayer',
            df,
            get_polygon='COORD',
            get_fill_color="""
            [
                255 * (SCORE >= 0.75) + 255 * (SCORE >= 0.5 && SCORE < 0.75) * (SCORE - 0.5) * 4,   
                255 * (SCORE < 0.5 ? SCORE * 2 : (1 - SCORE) * 4),                                  
                255 * (SCORE < 0.5 ? 1 - SCORE * 2 : 0),                                            
                180
            ]
            """,
            get_line_color='[80, 80, 80]',
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True
        )
    if type == 'Arc':
        df['COORD'] = df['COORD'].apply(polygon_centroid)
        df[['LONGITUDE_C', 'LATITUDE_C']] = pd.DataFrame(df['COORD'].tolist(), index=df.index)
        df['START'] = df['DEP_NAME'].map({
            '더현대서울': [126.928294, 37.526175],
            '신세계_강남': [127.004272, 37.506019],
            '롯데백화점_본점': [126.981664, 37.565183]
        })
        df[['LONGITUDE_S', 'LATITUDE_S']] = pd.DataFrame(df['START'].tolist(), index=df.index)
        df = df.sort_values('SCORE', ascending= False).head(10)

        layer = pdk.Layer(
            'ArcLayer',
            df,
            get_source_position = '[LONGITUDE_S, LATITUDE_S]',
            get_target_position = '[LONGITUDE_C, LATITUDE_C]',
            get_width = f'1 + 10 * SCORE',
            get_source_color='[255, 255, 120]',
            get_target_color='[255, 0, 0]',
            pickable=True,
            auto_highlight=True
            )

    tooltip = {
            "html": "<b>{ADDR_LV2} {ADDR_LV3}</b><br/>"
            f"지표: {{{score}}}<br/>"
            "지점: {DEP_NAME}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
            }
    
    center = [126.986, 37.565]
    
    view_state = pdk.ViewState(
        longitude=center[0],
        latitude=center[1],
        zoom=10,
        bearing=15,
        pitch=45
        )
    
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip= tooltip)
    chart = st.pydeck_chart(r)

    return chart

st.subheader("생활권 커버리지 지도")

col1, col2 = st.columns([4, 1])

with col2:
    option = st.radio(
        "카테고리 선택",
        options=["커버리지 지수", "추정 매출", "유입률"],
        index=0
    )

with col1:
    df_option = [metric, 'COVERAGE_SCORE', 'Poly'] if option == '커버리지 지수' else\
        [rci, 'SALES_ADDR', 'Poly'] if option == '추정 매출' else [rci, 'RATIO_TOT', 'Arc'] if option == '유입률' else None
    filtered_global = df_filter(df_option[0], dept, period, df_option[1])
    layer_vis(filtered_global , df_option[1], df_option[2])
    with st.expander("TOP 10 상세 정보"):
        filtered_global.sort_values(df_option[1], ascending = False, inplace = True)
        st.dataframe(filtered_global[['DEP_NAME','ADDR_LV2','ADDR_LV3', df_option[1]]].head(10))

# ---------------------------
# 하단 차트
# ---------------------------
col4, col5 = st.columns(2)

with col4:
    filtered = filtered.sort_values('STANDARD_YEAR_MONTH')
    months = filtered['STANDARD_YEAR_MONTH']
    sales = filtered['SALES_ADDR']
    mask = (
        (metric['DEP_NAME'] == dept) &
        (metric['STANDARD_YEAR_MONTH'].isin(months))
    )
    metric_avg = metric[mask].groupby('STANDARD_YEAR_MONTH')['SALES_ADDR'].median().reset_index()
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.subheader("📊 추정 매출 추이 (KRW, M : 백만)")

    fig1 = px.line(x = months, y = sales, markers=True, labels={'x' : '월', 'y' : '매출액'})
    fig1.add_scatter(
        x = metric_avg['STANDARD_YEAR_MONTH'],
        y = metric_avg['SALES_ADDR'],
        mode = 'lines+markers',
        name = '전체 평균',
        line = dict(color = 'gray', dash = 'dash'),
        marker = dict(symbol = 'circle', color= 'gray')
    )
    fig1.update_traces(name = '전체 평균', selector= dict(mode='lines+markers'))   
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    st.subheader("🎯 VIP 방문율 불렛 차트")
    st.caption("Target VIP = 유아가 있는 20~40대 여자")
    main_target = filtered['TARGET_VIP_RATE'].mean()
    main_vip = filtered['VIP_RATE'].mean()
    label_main = f"{dept} - {region} {dong}"

    mask_total = (
        (metric['DEP_NAME'] == dept) &
        (metric['STANDARD_YEAR_MONTH'] >= period[0]) &
        (metric['STANDARD_YEAR_MONTH'] <= period[1]) 
    )
    total = metric[mask_total]
    total_target = total['TARGET_VIP_RATE'].mean()
    total_vip = total['VIP_RATE'].mean()
    label_total = f"{dept} - 전체 지역"

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x = [main_vip * 100],
                          y = [label_main],
                          orientation= 'h',
                          width = 0.5,
                          marker_color='#2E8BC0',
                          name='VIP 비율',
                          hovertemplate='VIP 비율: %{x:.2f}%'
    ))
    fig2.add_trace(go.Scatter(
        x=[main_target * 100],
        y=[label_main],
        mode='markers',
        marker=dict(color='gray', size=22, symbol='line-ns-open'),
        name='TARGET VIP 비율',
        hovertemplate='Target VIP 비율: %{x:.2f}%'
    ))
    fig2.update_layout(
        title="선택 지역",
        xaxis_title="비율 (%)",
        yaxis_title="",
        barmode='overlay',
        template="simple_white",
        height=220,
        showlegend=True
    )
    st.plotly_chart(fig2)

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x = [total_vip * 100],
                          y = [label_total],
                          orientation= 'h',
                          width = 0.5,
                          marker_color='#B0C4DE',
                          name='VIP 비율',
                          hovertemplate='VIP 비율: %{x:.2f}%'
    ))
    fig3.add_trace(go.Scatter(
        x=[total_target * 100],
        y=[label_total],
        mode='markers',
        marker=dict(color='gray', size=22, symbol='line-ns-open'),
        name='TARGET VIP 비율',
        hovertemplate='Target VIP 비율: %{x:.2f}%'
    ))    
    fig3.update_layout(
        title="전체 지역",
        xaxis_title="비율 (%)",
        yaxis_title="",
        barmode='overlay',
        template="simple_white",
        height=220,
        showlegend=True
    )
    st.plotly_chart(fig3)
# ---------------------------
# 하단 상세 테이블
# ---------------------------
st.subheader("👥 지역 기반 소비자 세그먼트 & AI 마케팅 전략")

model = ChatOpenAI(model= "gpt-4o",temperature=0.7, api_key = api_key, cache = InMemoryCache())
name = region + " " + dong
z_df = zscored.select_dtypes('number')
point_lst = z_df.columns[z_df.loc[0] >= 1].tolist()
point = ', '.join(point_lst)
prompt = PromptTemplate(
    template=config["template"],
    input_variables=config["input_variables"]
    ).format(brand = dept, name = name, points = point)

def generate_response(prompt):
    result = model.invoke([
        SystemMessage(content = "당신은 유능한 백화점 마케팅 전문가입니다."),
        HumanMessage(content = prompt)
    ])
    st.info(result.content, icon="🤖")

col1, col2 = st.columns([2, 1])
with col1:
    with st.form("지역 마케팅 전략"):
        submitted = st.form_submit_button("전략 생성")
        if submitted:
            generate_response(prompt)
    # st.dataframe(customer_info, height = 400)

with col2:
    selected_cluster = customer_info.loc[customer_info['읍면동'] == dong, '클러스터'].values[0]
    cluster_counts = customer_info['클러스터'].value_counts().reset_index()
    cluster_counts.columns = ['클러스터', 'count']
    cluster_counts['비율'] = cluster_counts['count'] / cluster_counts['count'].sum() * 100
    base_colors = px.colors.qualitative.Plotly
    cluster_list = cluster_counts['클러스터'].tolist()
    color_map = {}
    for i, c in enumerate(cluster_list):
        base = base_colors[i % len(base_colors)]
        if c == selected_cluster:
            color_map[c] = base 
        else:
            rgb = px.colors.hex_to_rgb(base)
            color_map[c] = f'rgba({rgb[0]},{rgb[1]},{rgb[2]}, 0.2)'
    fig = px.treemap(
        cluster_counts,
        path=['클러스터'],
        values='비율',
        color='클러스터',
        color_discrete_map=color_map,
        title=f'지역 세그먼트 (selected : {dong})')
    fig.update_traces(
        texttemplate='<b>%{label}</b><br><span style="font-size:18px">%{value:.1f}%</span>',
    )
    fig.update_layout(
        margin=dict(t=40, l=10, r=10, b=10),
        height=400
    )
    st.plotly_chart(fig, use_container_width= True)

# ---------------------------
# 하단
# ---------------------------
st.markdown("---")
st.caption("🕒 Data updated: May 14, 2025")
