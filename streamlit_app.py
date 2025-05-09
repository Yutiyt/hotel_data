# app.py  ─────────────────────────────────────────────────────────────
import re, io, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx

# ░░ Streamlit page config ░░
st.set_page_config(page_title="Hotel Feedback Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ░░ Utility: cached data‑loader ░░
@st.cache_data
def load_and_prepare(file) -> tuple[pd.DataFrame, list]:
    EXP_COL  = ("Check-In Questionnaire: "
                "What persuaded you to choose our hotel for your stay?")
    SAT_COL  = ("Check-Out Questionnaire: "
                "What impressed you the most during your stay?")
    SPLIT    = re.compile(r"\s*;\s*")

    raw = pd.read_excel(file)

    def explode(df, col, label):
        return (
            df[["Customer", col]]
              .dropna()
              .assign(Feature=lambda x: x[col].apply(lambda s: SPLIT.split(str(s).strip())))
              .explode("Feature")
              .assign(Feature=lambda x: x["Feature"].str.strip(),
                      ResponseType=label)
              .drop(columns=[col])
        )
    tidy = pd.concat([explode(raw, EXP_COL, "Expect"),
                      explode(raw, SAT_COL, "Satisfy")])

    FEATURES = sorted(tidy["Feature"].unique())

    # build dummy matrix
    dummies = (tidy.assign(flag=True)
                     .pivot_table(index="Customer",
                                  columns=["ResponseType", "Feature"],
                                  values="flag",
                                  aggfunc="any",
                                  fill_value=False))
    dummies.columns = [f"{resp}_{feat}" for resp, feat in dummies.columns]
    df = dummies.reset_index()

    # Met / Missed / Surprise flags
    for f in FEATURES:
        df[f"Met_{f}"]      = df[f"Expect_{f}"] & df[f"Satisfy_{f}"]
        df[f"Missed_{f}"]   = df[f"Expect_{f}"] & ~df[f"Satisfy_{f}"]
        df[f"Surprise_{f}"] = ~df[f"Expect_{f}"] & df[f"Satisfy_{f}"]

    expect_cols   = [f"Expect_{f}"   for f in FEATURES]
    met_cols      = [f"Met_{f}"      for f in FEATURES]
    surprise_cols = [f"Surprise_{f}" for f in FEATURES]

    df["Expected_Count"]  = df[expect_cols].sum(axis=1)
    df["Met_Count"]       = df[met_cols].sum(axis=1)
    df["Surprise_Count"]  = df[surprise_cols].sum(axis=1)

    df["Satisfaction_Score"] = df["Met_Count"] / df["Expected_Count"].replace(0, np.nan)
    df["Surprise_Score"]     = df["Surprise_Count"] / len(FEATURES)

    return df, FEATURES

# ░░ Sidebar: File upload ░░
st.sidebar.header("Upload data")
data_file = st.sidebar.file_uploader("Raw_Data.xlsx", type=["xlsx"])

if not data_file:
    st.info("⬅️  Upload the **Raw_Data.xlsx** file to begin.")
    st.stop()

df, FEATURES = load_and_prepare(data_file)

# ░░ Tabs ░░
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🗃 Data", "📊 Expect vs Satisfy", "💎 Sat‑Surprise clusters",
     "🧭 4‑Area Map", "🔗 Corr‑Network", "🏷 Two‑Group Density"]
)

# ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Tidy preview & KPIs")
    st.write(df.head())
    st.metric("Guests", len(df))
    st.metric("Avg Satisfaction", f"{df['Satisfaction_Score'].mean():.2f}")

# ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Expectation vs Satisfaction frequency")
    feats = [c.replace("Expect_", "") for c in df.columns if c.startswith("Expect_")]
    exp_tot = df[[f"Expect_{f}" for f in feats]].sum()
    sat_tot = df[[f"Satisfy_{f}" for f in feats]].sum()

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(feats))
    ax.bar(x-0.2, exp_tot, 0.4, label="Expected")
    ax.bar(x+0.2, sat_tot, 0.4, label="Satisfied")
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=45, ha="right")
    ax.legend(); ax.set_ylabel("Guests"); plt.tight_layout()
    st.pyplot(fig)

# ────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Guest segments by Satisfaction & Surprise")
    valid = df[['Satisfaction_Score', 'Surprise_Score']].dropna()
    X_scaled = StandardScaler().fit_transform(valid)
    labels   = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(valid['Satisfaction_Score'], valid['Surprise_Score'],
                    c=labels, cmap="tab10")
    ax.set_xlabel("Satisfaction"); ax.set_ylabel("Surprise")
    ax.grid(True, ls=":")
    st.pyplot(fig)

# ────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Four‑quadrant expectation map")

    AREA_MAP = {
        "Room Comfort & Ambience": [
            "Comfortable and Clean Rooms",
            "Quiet and Restful Environment",
            "Stylish Interior Design",
        ],
        "Essential Convenience": [
            "Fast and Reliable Wi-Fi",
            "Easy Parking & Check-in",
            "Reservation & Communication",
        ],
        "Hospitality & Service": [
            "Friendly and Helpful Staff",
            "Family-Friendly Services",
        ],
        "Value‑Add Amenities": [
            "Modern Fitness Facilities",
            "Business Amenities",
            "Delicious Breakfast",
        ],
    }
    AREA_COORD = {
        "Room Comfort & Ambience": (+1, +1),
        "Essential Convenience":   (+1, -1),
        "Hospitality & Service":   (-1, +1),
        "Value‑Add Amenities":     (-1, -1),
    }

    if "Area_X" not in df.columns:
        xs, ys = [], []
        for _, row in df.iterrows():
            total = x_acc = y_acc = 0
            for area, feats in AREA_MAP.items():
                n = row[[f"Expect_{f}" for f in feats]].sum()
                if n:
                    axc, ayc = AREA_COORD[area]
                    x_acc += axc*n; y_acc += ayc*n; total += n
            xs.append(np.nan if total==0 else x_acc/total)
            ys.append(np.nan if total==0 else y_acc/total)
        df["Area_X"], df["Area_Y"] = xs, ys
        df["SatTier"] = np.where(df["Satisfaction_Score"]<0.7,
                                 "Low (<0.7)","High (≥0.7)")

    fig, ax = plt.subplots(figsize=(6,6))
    for tier,c in {"Low (<0.7)":"tab:blue","High (≥0.7)":"tab:orange"}.items():
        m = df["SatTier"]==tier
        ax.scatter(df.loc[m,"Area_X"], df.loc[m,"Area_Y"], c=c, label=tier,
                   s=60, edgecolors="k")
    ax.axhline(0, color="gray"); ax.axvline(0, color="gray")
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.2,1.2)
    ax.text( .6, .6,"Room\nComfort",ha="center")
    ax.text( .6,-.6,"Essential\nConv.",ha="center")
    ax.text(-.6, .6,"Hospitality\nService",ha="center")
    ax.text(-.6,-.6,"Value‑Add\nAmenities",ha="center")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(); ax.set_title("Colour = Satisfaction tier")
    st.pyplot(fig)

# ────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Correlation‑network")
    mode = st.selectbox("Mode", ["Expect", "Satisfy"])
    thresh = st.slider("|r| cut‑off", 0.10, 0.50, 0.25, 0.01)

    cols = [c for c in df.columns if c.startswith(f"{mode}_")]
    names= [c.replace(f"{mode}_","") for c in cols]
    corr = df[cols].astype(int).corr().values
    G = nx.Graph()
    for i,a in enumerate(names):
        for j in range(i+1,len(names)):
            w = corr[i,j]
            if abs(w)>=thresh:
                G.add_edge(a,names[j],weight=w)
    if len(G)==0:
        st.info("No edges above threshold.")
    else:
        pos = nx.spring_layout(G, seed=33)
        fig, ax = plt.subplots(figsize=(7,6))
        nx.draw_networkx_nodes(G,pos,node_color="lightgray",edgecolors="k",node_size=1000,ax=ax)
        nx.draw_networkx_edges(G,pos,
            width=[abs(G[u][v]['weight'])*4 for u,v in G.edges()],
            edge_color=['tab:orange' if G[u][v]['weight']>0 else 'tab:blue' for u,v in G.edges()],
            ax=ax)
        nx.draw_networkx_labels(G,pos,font_size=8,ax=ax)
        ax.set_axis_off()
        st.pyplot(fig)

# ────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Two‑macro‑group density")

    # ----- define the two macro‑groups ----------------------------
    GROUP_A = ["Comfortable and Clean Rooms",
               "Quiet and Restful Environment",
               "Modern Fitness Facilities",
               "Fast and Reliable Wi‑Fi"]

    GROUP_B = ["Reservation & Communication",
               "Family-Friendly Services",
               "Delicious Breakfast"]

    # ----- build lookup from the actual Expect_ columns -----------
    lookup = {c.replace("Expect_", ""): c
              for c in df.columns if c.startswith("Expect_")}

    # If the user hasn't uploaded a file yet, bail out gracefully
    if not lookup:
        st.info("Upload a data file first to see this chart.")
        st.stop()

    # ----- compute Macro_X safely (skip missing features) ---------
    if "Macro_X" not in df.columns:
        macro = []
        for _, row in df.iterrows():
            n_a = row[[lookup[f] for f in GROUP_A if f in lookup]].sum()
            n_b = row[[lookup[f] for f in GROUP_B if f in lookup]].sum()
            tot = n_a + n_b
            macro.append(np.nan if tot == 0 else (n_a - n_b) / tot)
        df["Macro_X"] = macro

    vals = df["Macro_X"].dropna()

    # ----- density + rug plot ------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.kdeplot(vals, fill=True, alpha=0.25, linewidth=0,
                clip=(-1.05, 1.05), color="gray", ax=ax)
    ax.scatter(vals, np.full_like(vals, -0.02),
               marker="|", color="black")

    ax.set_xlim(-1.05, 1.05);  ax.set_yticks([])
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["All B", "-lean B", "Balanced", "+lean A", "All A"])
    ax.set_xlabel("←  Family & Food Convenience      |      Comfort & Connectivity  →")
    ax.set_title("Distribution of guest expectations across the two macro‑groups")
    st.pyplot(fig)

    
