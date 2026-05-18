"""
원유 가격 × 장미 경매 × 기념일 영향 분석 대시보드
app.py와 독립적으로 운영 — 기존 flower_auction 테이블 불변
"""

import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from dotenv import load_dotenv

load_dotenv()

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="장미 가격 영향 분석 대시보드",
    page_icon="🌹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"]      { background: #fff0f3; }
[data-testid="stMetricValue"]  { font-size: 1.8rem; }
[data-testid="stMetricLabel"]  { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

KO_FONT = "Malgun Gothic, NanumGothic, AppleGothic, sans-serif"


# ── DB 연결 ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    try:
        url = st.secrets["database"]["url"]
    except Exception:
        url = os.getenv("DATABASE_URL")
    if not url:
        st.error("DATABASE_URL이 설정되지 않았습니다.")
        st.stop()
    return create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=5)


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="DB에서 데이터를 불러오는 중...")
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eng = get_engine()
    with eng.connect() as conn:
        oil = pd.read_sql(
            text("SELECT price_date, dubai FROM oil_price ORDER BY price_date"), conn
        )
        rose = pd.read_sql(
            text("""
                SELECT clcln_ymd, grad_nm,
                       top_utpc, lwet_utpc, avrg_utpc, tot_qyt, tot_amnt
                FROM rose_auction ORDER BY clcln_ymd
            """), conn
        )
        hol = pd.read_sql(
            text("SELECT holiday_date, holiday_name FROM holidays ORDER BY holiday_date"), conn
        )

    oil["price_date"]    = pd.to_datetime(oil["price_date"])
    rose["clcln_ymd"]    = pd.to_datetime(rose["clcln_ymd"])
    hol["holiday_date"]  = pd.to_datetime(hol["holiday_date"])
    return oil, rose, hol


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def monthly_agg(oil: pd.DataFrame, rose: pd.DataFrame) -> pd.DataFrame:
    o = oil.copy()
    o["ym"] = o["price_date"].dt.to_period("M")
    oil_m = o.groupby("ym")["dubai"].mean()

    r = rose.copy()
    r["ym"] = r["clcln_ymd"].dt.to_period("M")
    rose_m = r.groupby("ym")["avrg_utpc"].mean()

    df = pd.DataFrame({"rose": rose_m, "oil": oil_m}).dropna().sort_index()
    df.index = df.index.to_timestamp()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌹 분석 대시보드")
    st.divider()
    tab_sel = st.radio(
        "분석 탭 선택",
        [
            "📊 Tab 1: 유가 vs 장미 가격 (Overview)",
            "⏱️ Tab 2: 유가 시차 효과 (Granger)",
            "🎉 Tab 3: 기념일 이벤트 효과",
            "📦 Tab 4: 등급별 가격 분산 (ANOVA)",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("원유가 × 장미 경매 × 기념일 영향 분석")
    st.caption("데이터: 2025 화훼 경매 시세 · 두바이 원유")

st.title("🌹 원유·기념일이 장미 도매가에 미치는 영향 분석")
st.divider()

# 데이터 로드
try:
    oil_df, rose_df, hol_df = load_all()
except Exception as e:
    st.error(f"DB 연결 실패: {e}")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# Tab 1 : 유가 vs 장미 가격 기초 분석 (Overview)
# ═════════════════════════════════════════════════════════════════════════════
if "Tab 1" in tab_sel:
    st.subheader("📊 유가 vs 장미 가격 기초 분석")

    merged = monthly_agg(oil_df, rose_df)
    corr   = merged["oil"].corr(merged["rose"])

    # ── KPI ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("유가-장미 상관계수 (월별)", f"{corr:.3f}",
              delta="양의 상관" if corr > 0 else "음의 상관")
    c2.metric("분석 기간", f"{len(merged)}개월")
    c3.metric("장미 평균단가",  f"₩{rose_df['avrg_utpc'].mean():,.0f}")
    c4.metric("두바이 원유 평균", f"${oil_df['dubai'].mean():.1f}/bbl")

    # ── 이중 Y축 꺾은선 그래프 ─────────────────────────────────────────────
    view = st.radio("기간 단위", ["월별", "일별"], horizontal=True, key="t1_view")

    if view == "월별":
        x_oil   = merged.index
        y_oil   = merged["oil"]
        x_rose  = merged.index
        y_rose  = merged["rose"]
        title   = "두바이 원유가 vs 장미 평균단가 (월별 집계)"
    else:
        rose_d  = rose_df.groupby("clcln_ymd")["avrg_utpc"].mean().reset_index()
        x_oil   = oil_df["price_date"]
        y_oil   = oil_df["dubai"]
        x_rose  = rose_d["clcln_ymd"]
        y_rose  = rose_d["avrg_utpc"]
        title   = "두바이 원유가 vs 장미 평균단가 (일별)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_oil, y=y_oil,
        name="두바이 원유가 (USD/bbl)", yaxis="y1",
        line=dict(color="#e07b39", width=2),
        mode="lines+markers" if view == "월별" else "lines",
    ))
    fig.add_trace(go.Scatter(
        x=x_rose, y=y_rose,
        name="장미 평균단가 (원)", yaxis="y2",
        line=dict(color="#e91e8c", width=2),
        mode="lines+markers" if view == "월별" else "lines",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="날짜"),
        yaxis=dict(title="원유가 (USD/bbl)", color="#e07b39", side="left"),
        yaxis2=dict(title="장미 평균단가 (원)", color="#e91e8c",
                    side="right", overlaying="y"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        template="plotly_white",
        font=dict(family=KO_FONT),
        margin=dict(t=70, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"""
**인사이트**
- 월별 두바이 원유가와 장미 평균단가의 피어슨 상관계수는 **{corr:.3f}**입니다.
- 유가는 난방·운송비 등 생산 비용을 통해 화훼 도매가에 간접 영향을 미칩니다.
- 단순 상관계수는 인과관계를 설명하지 않으므로, **Tab 2**에서 Granger 시차 분석으로 보완하세요.
""")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2 : 유가 시차 효과 (Granger Causality)
# ═════════════════════════════════════════════════════════════════════════════
elif "Tab 2" in tab_sel:
    st.subheader("⏱️ 유가 시차 효과 분석 (Granger Causality)")

    merged = monthly_agg(oil_df, rose_df)

    # ── Lag별 피어슨 상관계수 ─────────────────────────────────────────────
    lag_rows = []
    for lag in range(4):
        tmp = pd.DataFrame({
            "rose": merged["rose"],
            "oil_lag": merged["oil"].shift(lag),
        }).dropna()
        r = tmp["rose"].corr(tmp["oil_lag"])
        lag_rows.append({"lag": lag, "label": f"Lag {lag}  (당월 -{lag}개월)", "r": r})
    lag_df = pd.DataFrame(lag_rows)
    best_lag = int(lag_df.loc[lag_df["r"].abs().idxmax(), "lag"])

    # ── Granger 검정 (1차 차분으로 정상화, RangeIndex로 변환해서 전달) ──
    diff = merged.diff().dropna().reset_index(drop=True)
    gc_rows = []
    try:
        gc = grangercausalitytests(diff[["rose", "oil"]], maxlag=3, verbose=False)
        for k, res in gc.items():
            f, p = res[0]["ssr_ftest"][:2]
            gc_rows.append({
                "시차": f"Lag {k}",
                "F-통계량": round(f, 3),
                "p-value":  round(p, 4),
                "유의 여부": "✅ 유의 (p<0.05)" if p < 0.05 else "❌ 비유의",
            })
    except Exception as e:
        st.warning(f"Granger 검정 오류 (데이터 부족 가능): {e}")

    # ── KPI ──────────────────────────────────────────────────────────────
    best_r = float(lag_df.loc[lag_df["lag"] == best_lag, "r"].iloc[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("최고 상관 시차", f"Lag {best_lag}개월")
    c2.metric("해당 시차 상관계수", f"{best_r:.3f}")
    if gc_rows:
        best_gc = min(gc_rows, key=lambda x: x["p-value"])
        c3.metric("Granger 최소 p-value", f"{best_gc['p-value']:.4f}",
                  delta=best_gc["유의 여부"])

    # ── Granger 결과 테이블 ───────────────────────────────────────────────
    if gc_rows:
        st.markdown("#### Granger Causality 검정 결과 (1차 차분 기준)")
        st.dataframe(pd.DataFrame(gc_rows), use_container_width=True, hide_index=True)

    # ── 막대 그래프: Lag별 상관계수 ──────────────────────────────────────
    st.markdown("#### Lag별 피어슨 상관계수")
    bar_colors = [
        "#e91e8c" if r["lag"] == best_lag else "#f4a7c3" for _, r in lag_df.iterrows()
    ]
    fig_lag = go.Figure(go.Bar(
        x=lag_df["label"],
        y=lag_df["r"],
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in lag_df["r"]],
        textposition="outside",
    ))
    fig_lag.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_lag.update_layout(
        title="유가 Lag별 장미 평균단가 피어슨 상관계수",
        xaxis_title="유가 시차",
        yaxis=dict(title="피어슨 상관계수", range=[-1, 1]),
        template="plotly_white",
        font=dict(family=KO_FONT),
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig_lag, use_container_width=True)

    # ── 최적 Lag 산점도 ───────────────────────────────────────────────────
    st.markdown(f"#### 최적 시차 (Lag {best_lag}개월) 산점도")
    scat = pd.DataFrame({
        "rose":    merged["rose"],
        "oil_lag": merged["oil"].shift(best_lag),
    }).dropna().reset_index()
    scat.columns = ["날짜", "장미 평균단가", f"두바이 원유가 (Lag {best_lag}개월 전)"]

    fig_sc = px.scatter(
        scat,
        x=f"두바이 원유가 (Lag {best_lag}개월 전)",
        y="장미 평균단가",
        trendline="ols",
        hover_data=["날짜"],
        template="plotly_white",
        color_discrete_sequence=["#e91e8c"],
    )
    fig_sc.update_layout(font=dict(family=KO_FONT), margin=dict(t=40, b=20),
                         yaxis_tickformat=",")
    st.plotly_chart(fig_sc, use_container_width=True)

    sig_lags = [r["시차"] for r in gc_rows if "✅" in r["유의 여부"]] if gc_rows else []
    st.info(f"""
**인사이트**
- Lag별 분석에서 **Lag {best_lag}개월** 시점의 유가와 장미 가격의 상관관계가 가장 높습니다 (r = {best_r:.3f}).
- Granger 인과성 검정은 과거 유가 정보가 장미 가격 예측에 추가적 설명력을 가지는지 검증합니다.
- {"유의한 시차: **" + ", ".join(sig_lags) + "**" if sig_lags else "분석 기간 내 Granger 인과성이 유의한 시차는 없습니다."} — 데이터 기간이 길수록 검정력이 높아집니다.
""")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3 : 기념일(이벤트) 효과 분석
# ═════════════════════════════════════════════════════════════════════════════
elif "Tab 3" in tab_sel:
    st.subheader("🎉 기념일 이벤트 효과 분석")

    # ── 분석 대상 기념일 필터 ─────────────────────────────────────────────
    hol_df["month"] = hol_df["holiday_date"].dt.month
    hol_df["day"]   = hol_df["holiday_date"].dt.day

    target = hol_df[
        (hol_df["month"] == 5) |   # 5월 가정의 달 전체
        (hol_df["day"] == 14)      # 매월 14일 기념일
    ].copy().reset_index(drop=True)

    if target.empty:
        st.warning("분석 대상 기념일 데이터가 없습니다.")
        st.stop()

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown(f"**분석 대상 기념일 — {len(target)}개**")
        st.dataframe(
            target[["holiday_date", "holiday_name"]].rename(
                columns={"holiday_date": "날짜", "holiday_name": "기념일명"}
            ),
            use_container_width=True, hide_index=True,
        )

    # ── 이벤트 윈도우 분석 (D-7 ~ 당일) ─────────────────────────────────
    rose_d = rose_df.groupby("clcln_ymd")["avrg_utpc"].mean().reset_index()
    rose_d.columns = ["date", "price"]

    rows = []
    for _, r in target.iterrows():
        h_date  = r["holiday_date"]
        h_name  = r["holiday_name"]
        w_start = h_date - pd.Timedelta(days=7)

        win  = rose_d[(rose_d["date"] >= w_start) & (rose_d["date"] <= h_date)]
        base = rose_d[rose_d["date"].dt.to_period("M") == h_date.to_period("M")]

        if len(win) < 2 or base.empty:
            continue

        ev_avg   = win["price"].mean()
        base_avg = base["price"].mean()
        premium  = (ev_avg - base_avg) / base_avg * 100 if base_avg > 0 else 0

        rows.append({
            "기념일":           h_name,
            "날짜":             h_date.strftime("%Y-%m-%d"),
            "이벤트 평균가(원)": round(ev_avg),
            "월 평시 평균가(원)": round(base_avg),
            "가격 프리미엄(%)":  round(premium, 2),
            "윈도우 데이터 수":  len(win),
        })

    if not rows:
        st.warning("이벤트 윈도우 기간의 장미 거래 데이터가 부족합니다.")
        st.stop()

    res = pd.DataFrame(rows).sort_values("가격 프리미엄(%)", ascending=False)

    with col_r:
        def _color_premium(val):
            color = "#d4edda" if val > 0 else ("#f8d7da" if val < 0 else "")
            return f"background-color: {color}"

        st.markdown("**기념일별 가격 프리미엄 분석표**")
        st.dataframe(
            res.style
               .map(_color_premium, subset=["가격 프리미엄(%)"])
               .format({
                   "이벤트 평균가(원)":  "₩{:,.0f}",
                   "월 평시 평균가(원)": "₩{:,.0f}",
                   "가격 프리미엄(%)":  "{:+.2f}%",
               }),
            use_container_width=True, hide_index=True,
        )

    # ── 막대 그래프: 가격 프리미엄 ───────────────────────────────────────
    st.markdown("#### 기념일별 장미 가격 프리미엄 (이벤트 윈도우 D-7 ~ 당일)")
    fig_ev = go.Figure(go.Bar(
        x=res["기념일"],
        y=res["가격 프리미엄(%)"],
        marker_color=["#e91e8c" if v >= 0 else "#5b8dee" for v in res["가격 프리미엄(%)"]],
        text=[f"{v:+.1f}%" for v in res["가격 프리미엄(%)"]],
        textposition="outside",
    ))
    fig_ev.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig_ev.update_layout(
        xaxis_title="기념일",
        yaxis_title="가격 프리미엄 (%)",
        xaxis_tickangle=-30,
        template="plotly_white",
        font=dict(family=KO_FONT),
        margin=dict(t=40, b=100),
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    # ── 기념일별 이벤트 윈도우 상세 시계열 ───────────────────────────────
    st.markdown("#### 이벤트 윈도우 상세 보기")
    sel_hol = st.selectbox("기념일 선택", res["기념일"].tolist(), key="hol_sel")
    sel_row  = res[res["기념일"] == sel_hol].iloc[0]
    sel_date = pd.to_datetime(sel_row["날짜"])

    ctx = rose_d[
        (rose_d["date"] >= sel_date - pd.Timedelta(days=14)) &
        (rose_d["date"] <= sel_date + pd.Timedelta(days=7))
    ]
    if not ctx.empty:
        fig_ctx = go.Figure()
        w0_str    = (sel_date - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        sel_str   = sel_date.strftime("%Y-%m-%d")
        fig_ctx.add_vrect(
            x0=w0_str, x1=sel_str,
            fillcolor="#ffe0ef", opacity=0.35,
            layer="below", line_width=0,
            annotation_text="이벤트 윈도우 (D-7 ~ D)",
            annotation_position="top left",
            annotation_font=dict(family=KO_FONT, size=12),
        )
        fig_ctx.add_trace(go.Scatter(
            x=ctx["date"], y=ctx["price"],
            mode="lines+markers",
            name="장미 평균단가",
            line=dict(color="#e91e8c", width=2.5),
            marker=dict(size=7),
        ))
        fig_ctx.add_shape(
            type="line", x0=sel_str, x1=sel_str, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(dash="dot", color="#c0005e", width=1.5),
        )
        fig_ctx.add_annotation(
            x=sel_str, y=1, xref="x", yref="paper",
            text=sel_hol, showarrow=False, yanchor="bottom",
            font=dict(family=KO_FONT, color="#c0005e", size=12),
        )
        fig_ctx.update_layout(
            title=f"{sel_hol} 전후 장미 평균단가 변화",
            xaxis_title="날짜",
            yaxis_title="평균단가 (원)",
            yaxis_tickformat=",",
            template="plotly_white",
            font=dict(family=KO_FONT),
            margin=dict(t=60, b=20),
        )
        st.plotly_chart(fig_ctx, use_container_width=True)

    top_hol  = res.iloc[0]["기념일"]
    top_prem = res.iloc[0]["가격 프리미엄(%)"]
    pos_cnt  = (res["가격 프리미엄(%)"] > 0).sum()
    st.info(f"""
**인사이트**
- 분석된 {len(res)}개 기념일 중 **{top_hol}**에서 가장 높은 프리미엄 **{top_prem:+.1f}%**가 관측됩니다.
- 전체 {len(res)}개 기념일 중 **{pos_cnt}개**({pos_cnt/len(res)*100:.0f}%)에서 양의 가격 프리미엄이 나타났습니다.
- 이벤트 윈도우(D-7 ~ 당일)는 화훼 유통 리드타임(수확 → 경매 → 소매)을 반영한 기간입니다.
- 음수 프리미엄은 해당 기념일 전후 출하량 증가로 오히려 가격이 낮아졌을 가능성을 시사합니다.
""")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 4 : 등급별 가격 분산 분석 (ANOVA)
# ═════════════════════════════════════════════════════════════════════════════
elif "Tab 4" in tab_sel:
    st.subheader("📦 등급별 가격 분산 분석 (One-Way ANOVA)")

    # ── 100개 이상 등급 필터 ─────────────────────────────────────────────
    grade_cnt   = rose_df["grad_nm"].value_counts()
    valid_grades = grade_cnt[grade_cnt >= 100].index.tolist()
    rf          = rose_df[rose_df["grad_nm"].isin(valid_grades)].copy()

    # 평균가 기준 내림차순 정렬
    grade_order = (
        rf.groupby("grad_nm")["avrg_utpc"].mean()
          .sort_values(ascending=False)
          .index.tolist()
    )

    summary = (
        rf.groupby("grad_nm")["avrg_utpc"]
          .agg(count="count", mean="mean", std="std", median="median")
          .reindex(grade_order)
          .reset_index()
          .rename(columns={
              "grad_nm": "등급",
              "count":   "데이터 수",
              "mean":    "평균가(원)",
              "std":     "표준편차",
              "median":  "중간값(원)",
          })
    )

    # ── One-Way ANOVA ────────────────────────────────────────────────────
    groups  = [rf[rf["grad_nm"] == g]["avrg_utpc"].dropna().values for g in grade_order]
    f_stat, p_val = stats.f_oneway(*groups)

    # ── KPI ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 등급 수",  f"{len(valid_grades)}개 (n≥100)")
    c2.metric("전체 샘플 수",  f"{len(rf):,}건")
    c3.metric("F-통계량",      f"{f_stat:,.2f}")
    c4.metric(
        "p-value",
        f"{p_val:.2e}",
        delta="유의 (p<0.05)" if p_val < 0.05 else "비유의",
        delta_color="normal" if p_val < 0.05 else "inverse",
    )

    # ── 요약 테이블 ───────────────────────────────────────────────────────
    def _color_grade(val):
        max_v = summary["평균가(원)"].max()
        min_v = summary["평균가(원)"].min()
        ratio = (val - min_v) / (max_v - min_v + 1e-9)
        r = int(255 - ratio * 80)
        g = int(240 - ratio * 160)
        b = int(255 - ratio * 60)
        return f"background-color: rgb({r},{g},{b})"

    st.markdown(f"**등급별 기술통계 ({len(valid_grades)}개 등급)**")
    st.dataframe(
        summary.style
               .map(_color_grade, subset=["평균가(원)"])
               .format({
                   "평균가(원)":  "₩{:,.0f}",
                   "표준편차":    "{:,.0f}",
                   "중간값(원)": "₩{:,.0f}",
                   "데이터 수":   "{:,}",
               }),
        use_container_width=True, hide_index=True,
    )

    # ── Boxplot ───────────────────────────────────────────────────────────
    st.markdown("#### 등급별 평균단가 분포 (Boxplot)")
    fig_box = px.box(
        rf, x="grad_nm", y="avrg_utpc",
        color="grad_nm",
        category_orders={"grad_nm": grade_order},
        labels={"grad_nm": "등급", "avrg_utpc": "평균단가 (원)"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        points="outliers",
    )
    fig_box.update_layout(
        showlegend=False,
        yaxis_tickformat=",",
        font=dict(family=KO_FONT),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ── 등급별 평균가 막대 그래프 ─────────────────────────────────────────
    st.markdown("#### 등급별 평균단가 비교")
    fig_bar = go.Figure(go.Bar(
        x=summary["등급"],
        y=summary["평균가(원)"],
        marker=dict(
            color=summary["평균가(원)"],
            colorscale="RdPu",
            showscale=False,
        ),
        error_y=dict(type="data", array=summary["표준편차"].tolist(), visible=True),
        text=[f"₩{v:,.0f}" for v in summary["평균가(원)"]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        xaxis_title="등급",
        yaxis=dict(title="평균단가 (원)", tickformat=","),
        template="plotly_white",
        font=dict(family=KO_FONT),
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    sig_yn = "**통계적으로 유의미합니다**" if p_val < 0.05 else "통계적으로 유의미하지 않습니다"
    top_g  = summary.iloc[0]["등급"]
    bot_g  = summary.iloc[-1]["등급"]
    top_v  = summary.iloc[0]["평균가(원)"]
    bot_v  = summary.iloc[-1]["평균가(원)"]
    st.info(f"""
**인사이트**
- One-Way ANOVA 결과: **F = {f_stat:,.2f}**, **p = {p_val:.2e}** → 등급 간 가격 차이는 {sig_yn} (α = 0.05).
- 가장 높은 평균단가 등급은 **{top_g}** (₩{top_v:,.0f}), 가장 낮은 등급은 **{bot_g}** (₩{bot_v:,.0f})입니다.
- F-통계량이 클수록 등급 간 분산이 등급 내 분산보다 크다는 의미로, 등급이 가격 결정의 핵심 변수임을 시사합니다.
- 추가로 Tukey HSD 사후 검정을 수행하면 어느 등급 쌍 간 차이가 유의미한지 파악할 수 있습니다.
""")
