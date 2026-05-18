import streamlit as st

pg = st.navigation(
    {
        "화훼 경매": [
            st.Page("pages/1_화훼_경매_시세.py",  title="화훼 경매 시세",  icon="🌸"),
        ],
        "장미 분석": [
            st.Page("pages/2_Rose_Analysis.py", title="장미 영향 분석", icon="🌹"),
        ],
    },
    position="sidebar",
)
pg.run()
