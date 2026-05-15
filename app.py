"""
화훼 경매 시세 대시보드
Streamlit + Supabase(PostgreSQL) + Plotly
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import date, timedelta

# ── 페이지 기본 설정 ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="화훼 경매 시세 대시보드",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2rem; }
    [data-testid="stSidebar"] { background: #f8f4f0; }
    div[data-testid="stRadio"] label { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── DB 연결 ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    url = st.secrets["database"]["url"]
    return create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=5)


# ── 데이터 날짜 범위 (DB 기준) ────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_date_bounds() -> tuple[date, date]:
    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT MIN(clcln_ymd)::date, MAX(clcln_ymd)::date FROM flower_auction")
        ).fetchone()
    return row[0], row[1]


# ── 화훼부류 목록 ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_category_list() -> list[str]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT flwp_se_nm FROM flower_auction ORDER BY flwp_se_nm")
        ).fetchall()
    return [r[0] for r in rows]


# ── 품목 목록 (화훼부류 연동) ─────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_product_list(category: str | None) -> list[str]:
    if category:
        query = text(
            "SELECT DISTINCT pdlt_nm FROM flower_auction "
            "WHERE flwp_se_nm = :cat ORDER BY pdlt_nm"
        )
        params = {"cat": category}
    else:
        query = text("SELECT DISTINCT pdlt_nm FROM flower_auction ORDER BY pdlt_nm")
        params = {}
    with get_engine().connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [r[0] for r in rows]


# ── 데이터 조회 (LIMIT 없음 — 기간 전체) ─────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="데이터 조회 중...")
def fetch_data(
    category: str | None,
    products: tuple[str, ...],
    date_from: date,
    date_to: date,
) -> pd.DataFrame:
    conditions = ["clcln_ymd BETWEEN :d_from AND :d_to"]
    params: dict = {"d_from": date_from, "d_to": date_to}

    if category:
        conditions.append("flwp_se_nm = :cat")
        params["cat"] = category

    if products:
        placeholders = ", ".join(f":p{i}" for i in range(len(products)))
        conditions.append(f"pdlt_nm IN ({placeholders})")
        for i, p in enumerate(products):
            params[f"p{i}"] = p

    where = " AND ".join(conditions)
    query = text(
        f"""
        SELECT clcln_ymd, flwp_se_nm, pdlt_nm, spcs_nm, grad_nm,
               top_utpc, lwet_utpc, avrg_utpc, tot_qyt, tot_amnt
        FROM flower_auction
        WHERE {where}
        ORDER BY clcln_ymd DESC
        """
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    df["clcln_ymd"] = pd.to_datetime(df["clcln_ymd"])
    return df


# ── KPI 계산 ─────────────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> dict:
    latest_date = df["clcln_ymd"].max()
    latest = df[df["clcln_ymd"] == latest_date]
    prev_dates = df[df["clcln_ymd"] < latest_date]["clcln_ymd"]
    prev = df[df["clcln_ymd"] == prev_dates.max()] if not prev_dates.empty else pd.DataFrame()
    avg_now  = latest["avrg_utpc"].mean()
    avg_prev = prev["avrg_utpc"].mean() if not prev.empty else None
    return {
        "avg_price":   avg_now,
        "avg_delta":   avg_now - avg_prev if avg_prev is not None else None,
        "top_price":   df["top_utpc"].max(),
        "total_qty":   df["tot_qyt"].sum(),
        "total_amnt":  df["tot_amnt"].sum(),
        "latest_date": latest_date,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Simple_flower.svg/240px-Simple_flower.svg.png",
        width=60,
    )
    st.title("필터 설정")
    st.divider()

    try:
        category_list  = fetch_category_list()
        data_min, data_max = fetch_date_bounds()
    except Exception as e:
        st.error(f"DB 연결 실패: {e}")
        st.info("`.streamlit/secrets.toml`의 DATABASE_URL을 확인하세요.")
        st.stop()

    # ① 화훼부류
    st.markdown("**화훼부류**")
    selected_category = st.selectbox(
        "화훼부류",
        ["전체"] + category_list,
        label_visibility="collapsed",
    )
    category_param = None if selected_category == "전체" else selected_category

    # ② 품목명 (화훼부류 연동, 자동선택 없음)
    st.markdown("**품목명**")
    product_list = fetch_product_list(category_param)

    search_term = st.text_input(
        "품목 검색", placeholder="예) 장미, 국화...", label_visibility="collapsed"
    )
    filtered_products = (
        [p for p in product_list if search_term in p] if search_term else product_list
    )

    selected_products = st.multiselect(
        "품목명 선택 (복수 가능)",
        filtered_products,
        default=[],                 # 자동선택 없음
        label_visibility="collapsed",
    )

    st.divider()

    # ③ 조회 기간
    st.markdown("**조회 기간**")

    PERIOD_PRESETS = {
        "일주일": 7,
        "한달":   30,
        "분기":   90,
        "반기":   180,
        "년간":   365,
        "직접입력": None,
    }

    selected_preset = st.radio(
        "기간 프리셋",
        list(PERIOD_PRESETS.keys()),
        index=0,                    # 기본값: 일주일
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_preset == "직접입력":
        custom_range = st.date_input(
            "날짜 범위",
            value=(data_max - timedelta(days=7), data_max),
            min_value=data_min,
            max_value=data_max,
            format="YYYY/MM/DD",
            label_visibility="collapsed",
        )
        if isinstance(custom_range, (list, tuple)) and len(custom_range) == 2:
            date_from, date_to = custom_range
        else:
            st.info("종료 날짜를 선택해 주세요.")
            st.stop()
    else:
        days = PERIOD_PRESETS[selected_preset]
        date_to   = data_max
        date_from = max(data_min, data_max - timedelta(days=days))

    st.caption(f"{date_from.strftime('%Y.%m.%d')} ~ {date_to.strftime('%Y.%m.%d')}")

    st.divider()
    st.caption("데이터: 2025년 화훼 경매 시세")
    st.caption("출처: 농림축산식품부")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 영역
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌸 화훼 경매 시세 대시보드")

# 품목 미선택 시 안내 화면
if not selected_products:
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; padding: 80px 0; color: #999;">
            <div style="font-size:4rem;">🌷</div>
            <div style="font-size:1.3rem; margin-top:16px;">
                사이드바에서 <strong>화훼부류</strong>와 <strong>품목명</strong>을 선택하면<br>
                경매 시세 데이터가 표시됩니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# 선택 정보 캡션
st.caption(
    f"화훼부류: **{selected_category}** | "
    f"품목: **{', '.join(selected_products)}** | "
    f"기간: **{date_from.strftime('%Y.%m.%d')} ~ {date_to.strftime('%Y.%m.%d')}**"
)

# 데이터 로드
df = fetch_data(category_param, tuple(selected_products), date_from, date_to)

if df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다. 필터를 변경해 보세요.")
    st.stop()

# ── KPI 카드 (4열) ───────────────────────────────────────────────────────────
kpis = compute_kpis(df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    delta_str = f"₩{kpis['avg_delta']:+,.0f}" if kpis["avg_delta"] is not None else None
    st.metric(
        label=f"최근 평균가 ({kpis['latest_date'].strftime('%m/%d')})",
        value=f"₩{kpis['avg_price']:,.0f}",
        delta=delta_str,
    )
with col2:
    st.metric(label="기간 최고가", value=f"₩{kpis['top_price']:,.0f}")
with col3:
    st.metric(label="누적 거래 물량", value=f"{kpis['total_qty']:,.0f} 단")
with col4:
    st.metric(label="누적 거래 금액", value=f"₩{kpis['total_amnt'] / 1_000_000:,.1f}M")

st.divider()

# ── 날짜별 평균가 추이 (품목별 다중 라인) ────────────────────────────────────
st.subheader("📈 날짜별 평균가 추이")

trend = (
    df.groupby(["clcln_ymd", "pdlt_nm"], as_index=False)
    .agg(avg_price=("avrg_utpc", "mean"), tot_qty=("tot_qyt", "sum"))
    .sort_values("clcln_ymd")
)

fig_line = px.line(
    trend,
    x="clcln_ymd",
    y="avg_price",
    color="pdlt_nm",
    markers=len(trend) <= 300,
    labels={"clcln_ymd": "날짜", "avg_price": "평균가 (원)", "pdlt_nm": "품목"},
    template="plotly_white",
)
fig_line.update_traces(line_width=2.5, marker_size=5)
fig_line.update_layout(
    hovermode="x unified",
    xaxis_title=None,
    yaxis_tickformat=",",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=40, b=0),
)
st.plotly_chart(fig_line, use_container_width=True)

# ── 품목별 물량 비중 + 등급별 분포 (2열) ─────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🥧 품목별 거래 물량 비중")
    pie_data = df.groupby("pdlt_nm", as_index=False)["tot_qyt"].sum()
    fig_pie = px.pie(
        pie_data,
        names="pdlt_nm",
        values="tot_qyt",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("📊 등급별 평균가 분포")
    grade_order = (
        df.groupby("grad_nm")["avrg_utpc"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig_box = px.box(
        df,
        x="grad_nm",
        y="avrg_utpc",
        color="grad_nm",
        category_orders={"grad_nm": grade_order},
        labels={"grad_nm": "등급", "avrg_utpc": "평균가 (원)"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_box.update_layout(showlegend=False, yaxis_tickformat=",", margin=dict(t=20, b=0))
    st.plotly_chart(fig_box, use_container_width=True)

# ── 상세 데이터 테이블 ────────────────────────────────────────────────────────
st.subheader("📋 상세 내역")

display_df = df.copy()
display_df["clcln_ymd"] = display_df["clcln_ymd"].dt.strftime("%Y-%m-%d")
display_df = display_df.rename(columns={
    "clcln_ymd":  "정산일자",
    "flwp_se_nm": "화훼부류",
    "pdlt_nm":    "품목명",
    "spcs_nm":    "품종명",
    "grad_nm":    "등급",
    "top_utpc":   "최고가(원)",
    "lwet_utpc":  "최저가(원)",
    "avrg_utpc":  "평균가(원)",
    "tot_qyt":    "총물량(단)",
    "tot_amnt":   "총금액(원)",
})

st.dataframe(
    display_df,
    use_container_width=True,
    height=380,
    column_config={
        "최고가(원)": st.column_config.NumberColumn(format="₩%,.0f"),
        "최저가(원)": st.column_config.NumberColumn(format="₩%,.0f"),
        "평균가(원)": st.column_config.NumberColumn(format="₩%,.0f"),
        "총금액(원)": st.column_config.NumberColumn(format="₩%,.0f"),
        "총물량(단)": st.column_config.NumberColumn(format="%,.0f"),
    },
    hide_index=True,
)
