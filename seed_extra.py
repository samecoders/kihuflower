"""
holy.csv / oil.csv / rose.csv → Supabase(PostgreSQL) 적재 스크립트

테이블:
  holidays     ← holy.csv   (날짜, 기념일)
  oil_price    ← oil.csv    (날짜, 두바이 원유가)
  rose_auction ← rose.csv   (화훼 경매 - 장미 전용)

실행:
  python seed_extra.py
"""

import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError(".env 파일에 DATABASE_URL이 없습니다.")

CHUNK_SIZE = 5_000


# ── CSV 로더 ─────────────────────────────────────────────────────────────────

def load_holidays(path: str) -> pd.DataFrame:
    """holy.csv: 날짜(YYYYMMDD), 기념일명"""
    df = pd.read_csv(path, encoding="cp949")
    df.columns = ["holiday_date", "holiday_name"]
    df["holiday_date"] = pd.to_datetime(df["holiday_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["holiday_date"])
    return df


def load_oil_price(path: str) -> pd.DataFrame:
    """oil.csv: 기간(YYYYMMDD), Dubai(USD/배럴)"""
    df = pd.read_csv(path, encoding="cp949")
    df.columns = ["price_date", "dubai"]
    df["price_date"] = pd.to_datetime(df["price_date"], format="%Y%m%d", errors="coerce")
    df["dubai"] = pd.to_numeric(df["dubai"], errors="coerce")
    df = df.dropna(subset=["price_date", "dubai"])
    return df


def load_rose_auction(path: str) -> pd.DataFrame:
    """rose.csv: 화훼 경매 데이터 (flower_auction과 동일 스키마)"""
    df = pd.read_csv(
        path,
        encoding="cp949",
        dtype={
            "CLCLN_YMD": str,
            "FLWP_SE_NM": str,
            "PDLT_NM":    str,
            "SPCS_NM":    str,
            "GRAD_NM":    str,
            "TOP_UTPC":   float,
            "LWET_UTPC":  float,
            "AVRG_UTPC":  float,
            "TOT_QYT":    float,
            "TOT_AMNT":   float,
        },
    )
    df.columns = df.columns.str.lower()
    df["clcln_ymd"] = pd.to_datetime(df["clcln_ymd"], format="%Y%m%d", errors="coerce")
    num_cols = ["top_utpc", "lwet_utpc", "avrg_utpc", "tot_qyt", "tot_amnt"]
    df[num_cols] = df[num_cols].fillna(0)
    return df


# ── 공통 업로드 ───────────────────────────────────────────────────────────────

def upload(df: pd.DataFrame, engine, table: str) -> None:
    print(f"  적재 중: {table} ({len(df):,}행)")
    t0 = time.time()
    df.to_sql(
        name=table,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=CHUNK_SIZE,
        method="multi",
    )
    print(f"  완료: {len(df):,}행 ({time.time() - t0:.1f}초)")


# ── 인덱스 ────────────────────────────────────────────────────────────────────

def create_indexes(engine) -> None:
    indexes = [
        # holidays
        "CREATE INDEX IF NOT EXISTS idx_holidays_date ON holidays (holiday_date);",
        # oil_price
        "CREATE INDEX IF NOT EXISTS idx_oil_price_date ON oil_price (price_date);",
        # rose_auction
        "CREATE INDEX IF NOT EXISTS idx_rose_pdlt_nm   ON rose_auction (pdlt_nm);",
        "CREATE INDEX IF NOT EXISTS idx_rose_clcln_ymd ON rose_auction (clcln_ymd);",
        "CREATE INDEX IF NOT EXISTS idx_rose_pdlt_date ON rose_auction (pdlt_nm, clcln_ymd);",
    ]
    with engine.connect() as conn:
        for sql in indexes:
            conn.execute(text(sql))
        conn.commit()
    print("  인덱스 생성 완료")


# ── 검증 ─────────────────────────────────────────────────────────────────────

def verify(engine) -> None:
    queries = {
        "holidays":     "SELECT COUNT(*) FROM holidays",
        "oil_price":    "SELECT COUNT(*) FROM oil_price",
        "rose_auction": "SELECT COUNT(*) FROM rose_auction",
    }
    with engine.connect() as conn:
        for table, q in queries.items():
            count = conn.execute(text(q)).scalar()
            print(f"  {table}: {count:,}행")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    jobs = [
        ("holy.csv",  load_holidays,     "holidays"),
        ("oil.csv",   load_oil_price,    "oil_price"),
        ("rose.csv",  load_rose_auction, "rose_auction"),
    ]

    for csv_file, loader, table in jobs:
        path = Path(csv_file)
        if not path.exists():
            print(f"[건너뜀] 파일 없음: {csv_file}")
            continue
        print(f"\n[{table}] {csv_file} 로드 중...")
        df = loader(str(path))
        print(f"  → {len(df):,}행 × {len(df.columns)}열 로드 완료")
        upload(df, engine, table)

    print("\n[인덱스 생성]")
    create_indexes(engine)

    print("\n[적재 결과 검증]")
    verify(engine)

    print("\n모든 적재가 완료되었습니다.")


if __name__ == "__main__":
    main()
