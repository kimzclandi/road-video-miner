"""All numbers come from immutable experiment evidence."""

from pathlib import Path

import pandas as pd
import streamlit as st

from miner import read

ROOT = Path(__file__).resolve().parent
st.title("Road Video Miner · 视频片段挖掘")
st.caption("真实KITTI序列 · 等两秒预算 · 无训练 · 非官方KITTI指标")
if not (ROOT / "reports/result.json").exists():
    st.info("实验尚未完成；不展示模拟数字。")
    st.stop()
r = read(ROOT / "reports/result.json")
st.metric("时序策略 − 随机 · 参考漏检轨迹覆盖", f"{100 * r['primary_delta']:+.2f} 个百分点")
st.write(
    f"序列bootstrap 95%区间：[{100 * r['ci95'][0]:.2f}, {100 * r['ci95'][1]:.2f}]；有效序列 {r['supported_sequences']}/{r['evaluation_sequences']}。"
)
st.warning("覆盖更多参考失败不等于改善检测模型。5Hz观测也可能漏掉快速事件。")
split = st.selectbox("隔离集合", ["eval", "dev", "pool"])
df = pd.DataFrame(r["rows"])
df = df[df["split"] == split]
st.dataframe(
    df.groupby("method")[
        ["failure_coverage", "track_coverage", "redundant_track_fraction", "appearance_cosine", "seconds"]
    ].mean()
)
seq = st.selectbox("序列", sorted(df["sequence"].unique()))
st.dataframe(df[df["sequence"] == seq], hide_index=True)
with st.expander("原始参考、协议与边界"):
    st.json(r["references"][seq])
    st.json(read(ROOT / "reports/protocol.json"))
