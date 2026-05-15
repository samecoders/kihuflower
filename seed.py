"""
화훼 경매 데이터 → Supabase(PostgreSQL) 고속 적재 스크립트

실행 전 준비:
  pip install pandas sqlalchemy psycopg2-binary python-dotenv
  .env 파일에 DATABASE_URL 설정 (또는 직접 수정)

사용법:
  python seed.py
  python seed.py --file "다른파일.csv"  --chunk 10000
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────────────────
DEFAULT_CSV = "2025년화훼 경매 시세.csv"
TABLE_NAME  = "flower_auction"
CHUNK_SIZE  = 5_000

# .env 또는 아래 직접 입력 (Supabase → Settings → Database → Connection string)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:<PASSWORD>@db.<PROJECT-REF>.supabase.co:5432/postgres"
)
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    """CSV를 읽고 컬럼·타입을 정규화한다."""
    print(f"[1/4] CSV 로드 중: {path}")
    df = pd.read_csv(
        path,
        encoding="cp949",        # 한글 Windows 기본 인코딩
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

    # 컬럼명 소문자 통일 (PostgreSQL 관례)
    df.columns = df.columns.str.lower()

    # 정산일자 → date 타입
    df["clcln_ymd"] = pd.to_datetime(df["clcln_ymd"], format="%Y%m%d", errors="coerce")

    # 숫자 컬럼 결측치 0 처리
    num_cols = ["top_utpc", "lwet_utpc", "avrg_utpc", "tot_qyt", "tot_amnt"]
    df[num_cols] = df[num_cols].fillna(0)

    print(f"    → 로드 완료: {len(df):,}행 × {len(df.columns)}열")
    return df


def upload(df: pd.DataFrame, engine, chunk: int) -> None:
    """청크 단위 bulk insert."""
    print(f"[2/4] DB 적재 시작 (청크={chunk:,})")
    t0 = time.time()

    df.to_sql(
        name=TABLE_NAME,
        con=engine,
        if_exists="replace",   # 재실행 시 테이블 재생성
        index=False,
        chunksize=chunk,
        method="multi",        # executemany 대신 multi-row VALUES — 3~5배 빠름
    )

    elapsed = time.time() - t0
    print(f"    → 적재 완료: {len(df):,}행 ({elapsed:.1f}초)")


def create_indexes(engine) -> None:
    """조회 빈도가 높은 컬럼에 B-Tree 인덱스 생성."""
    print("[3/4] 인덱스 생성 중...")
    indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_pdlt_nm  ON {TABLE_NAME} (pdlt_nm);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_clcln_ymd ON {TABLE_NAME} (clcln_ymd);",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_flwp      ON {TABLE_NAME} (flwp_se_nm);",
        # 복합 인덱스: 품목 + 날짜 조합 쿼리 최적화
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_pdlt_date ON {TABLE_NAME} (pdlt_nm, clcln_ymd);",
    ]
    with engine.connect() as conn:
        for sql in indexes:
            conn.execute(text(sql))
            print(f"    ✓ {sql.split('EXISTS')[1].split(' ON')[0].strip()}")
        conn.commit()
    print("    → 인덱스 생성 완료")


def verify(engine) -> None:
    """적재 결과 간단 검증."""
    print("[4/4] 적재 결과 검증...")
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
        sample = conn.execute(
            text(f"SELECT pdlt_nm, clcln_ymd, avrg_utpc FROM {TABLE_NAME} LIMIT 3")
        ).fetchall()
    print(f"    총 {count:,}행 확인")
    for row in sample:
        print(f"    {row}")


def main():
    parser = argparse.ArgumentParser(description="화훼 경매 데이터 Supabase 적재")
    parser.add_argument("--file",  default=DEFAULT_CSV, help="CSV 파일 경로")
    parser.add_argument("--chunk", type=int, default=CHUNK_SIZE, help="청크 크기")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    df = load_csv(str(csv_path))
    upload(df, engine, args.chunk)
    create_indexes(engine)
    verify(engine)

    print("\n모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main()
