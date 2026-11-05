"""
Browse the parquet files under data/ in a web page.

    streamlit run explore.py
"""

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from pathlib import Path

DATA = Path("data")
st.set_page_config(page_title="CS486 data", layout="wide")

files = sorted(DATA.rglob("*.parquet"))
if not files:
    st.error("No parquet files under data/")
    st.stop()

pick = st.sidebar.selectbox("file", [str(f.relative_to(DATA)) for f in files])
nrows = st.sidebar.slider("rows", 5, 200, 20)
path = DATA / pick


@st.cache_data
def peek(p, n):
    return next(pq.ParquetFile(p).iter_batches(batch_size=n)).to_pandas()


meta = pq.ParquetFile(path).metadata

c1, c2, c3 = st.columns(3)
c1.metric("rows", f"{meta.num_rows:,}")
c2.metric("columns", meta.num_columns)
c3.metric("file size", f"{path.stat().st_size / 1e6:,.0f} MB")

st.subheader(pick)
df = peek(path, nrows)
st.dataframe(df, use_container_width=True)

st.download_button("download these rows as CSV",
                   df.to_csv(index=False).encode(),
                   file_name=f"head_{path.stem}.csv")

with st.expander("column summary"):
    s = peek(path, 200_000)
    st.caption(f"based on the first {len(s):,} rows")
    st.dataframe(pd.DataFrame({
        "type": s.dtypes.astype(str),
        "filled %": (s.notna().mean() * 100).round(1),
        "unique": s.nunique(),
        "example": [str(s[c].dropna().iloc[0]) if s[c].notna().any() else ""
                    for c in s.columns],
    }), use_container_width=True)