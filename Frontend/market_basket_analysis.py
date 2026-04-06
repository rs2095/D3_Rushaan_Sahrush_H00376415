import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import hashlib
from audit_logger import log_event
from Managers.MarketBasketManager import MarketBasketEngine
from styles import inject_styles


def show():
    st.set_page_config(page_title="Market Basket Analysis", layout="wide")
    inject_styles()

    st.title("Market Basket Analysis")
    st.markdown("FP-Growth Powered Cross-Selling Engine")
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Orders Dataset (Must contain: Customer_ID, Department_Name)",
        type=["csv"],
        key="market_basket_upload"
    )

    if not uploaded_file:
        st.session_state.pop("last_uploaded_hash", None)
        st.info("Upload a dataset to begin market basket analysis.")
        st.stop()

    #computes file hash and logs upload event
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if "last_uploaded_hash" not in st.session_state:
        st.session_state.last_uploaded_hash = None

    if st.session_state.last_uploaded_hash != file_hash:
        log_event(action_type="upload", page_name="Market Basket Intelligence", file_name=uploaded_file.name)
        st.session_state.last_uploaded_hash = file_hash

    #reads and validates uploaded csv
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file as a CSV: {e}")
        st.stop()

    if df.empty:
        st.error("The uploaded file is empty. Please upload a valid dataset.")
        st.stop()

    required_cols = {"Customer_ID", "Department_Name"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing required column(s): **{', '.join(missing)}**. Please check your file and re-upload.")
        st.stop()

    df = df.dropna(subset=list(required_cols))
    if df.empty:
        st.error("No valid rows remain after removing missing values in Customer_ID and Department_Name.")
        st.stop()

    st.success(f"{len(df):,} records loaded")

    #threshold controls for rule generation
    st.markdown("### Rule Threshold Configuration")
    col1, col2, col3 = st.columns(3)
    min_support    = col1.slider("Minimum Support",    0.001, 0.1,  0.01,  0.001)
    min_confidence = col2.slider("Minimum Confidence", 0.1,   1.0,  0.3,   0.05)
    min_lift       = col3.slider("Minimum Lift",       1.0,   5.0,  1.2,   0.1)

    run_analysis = st.button("Run Market Basket Analysis")
    if not run_analysis:
        st.stop()

    #runs fp-growth engine and generates association rules
    with st.spinner("Running FP-Growth..."):
        try:
            engine = MarketBasketEngine(
                min_support=min_support,
                min_confidence=min_confidence,
                min_lift=min_lift
            )
            result = engine.generate_rules(df)

            if isinstance(result, (list, tuple)):
                if len(result) == 2:
                    rules, basket = result
                    frequent_itemsets = None
                elif len(result) == 3:
                    rules, frequent_itemsets, basket = result
                else:
                    st.error(f"Unexpected number of return values from generate_rules ({len(result)}). Contact your developer.")
                    st.stop()
            else:
                st.error("generate_rules returned an unexpected type. Contact your developer.")
                st.stop()

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

        log_event(action_type="run_analysis", page_name="Market Basket Intelligence",
                  details=f"Support={min_support}, Confidence={min_confidence}, Lift={min_lift}")

    if rules is None or (hasattr(rules, 'empty') and rules.empty):
        st.warning("No association rules found with current thresholds. Try lowering support or confidence.")
        st.stop()

    #displays association insight kpis
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Association Insights and Metrics")

    kpi_cols = st.columns(4)
    kpi_data = [
        ("Total Rules",     f"{len(rules):,}",                    "After filtering"),
        ("Avg Lift",        f"{rules['lift'].mean():.2f}",         "Association strength"),
        ("Avg Confidence",  f"{rules['confidence'].mean():.2f}",   "Predictive strength"),
        ("Avg Support",     f"{rules['support'].mean():.3f}",      "Average occurrence"),
    ]
    for col, (title, value, sub) in zip(kpi_cols, kpi_data):
        col.markdown(f"""
            <div class="market-card border-blue-500">
                <div class="market-kpi-title">{title}</div>
                <div class="market-kpi-value">{value}</div>
                <div class="market-kpi-sub text-gray-500">{sub}</div>
            </div>
        """, unsafe_allow_html=True)

    #displays top 10 cross-selling rules
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### Top 10 Strongest Cross-Selling Rules")

    top_rules = rules.sort_values(by="lift", ascending=False).head(10)
    for _, row in top_rules.iterrows():
        st.markdown(f"""
            <div class="market-card border-blue-500">
                <div class="market-kpi-value">{row['antecedents']} → {row['consequents']}</div>
                <div class="market-kpi-sub text-gray-500">
                    Support: {row['support']:.3f} | Confidence: {row['confidence']:.3f} | Lift: {row['lift']:.3f}
                </div>
            </div><br>
        """, unsafe_allow_html=True)

    #renders rule network graph
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### Rule Network")

    try:
        G = nx.DiGraph()
        for _, row in top_rules.iterrows():
            antecedents = str(row['antecedents']).split(', ')
            consequents = str(row['consequents']).split(', ')
            for a in antecedents:
                for c in consequents:
                    G.add_edge(a, c, weight=row['lift'])

        if G.number_of_nodes() == 0:
            st.info("Not enough rules to build a network graph.")
        else:
            pos = nx.spring_layout(G, seed=42)
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]; x1, y1 = pos[v]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#888'), hoverinfo='none', mode='lines')
            node_x, node_y, node_text = [], [], []
            for node in G.nodes():
                x, y = pos[node]; node_x.append(x); node_y.append(y); node_text.append(node)

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text', text=node_text,
                textposition="top center", hoverinfo='text',
                marker=dict(
                    showscale=True, colorscale='YlGnBu',
                    color=[sum(d['weight'] for u, v, d in G.edges(data=True) if v == node) for node in G.nodes()],
                    size=30, colorbar=dict(thickness=15, title='Incoming Lift Sum'), line_width=2
                )
            )
            fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            ))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render network graph: {e}")

    #displays top 500 rules table with download
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### Top 500 Rules by Lift")

    top_rules_preview = rules.sort_values(by="lift", ascending=False).head(500).reset_index(drop=True)
    st.dataframe(top_rules_preview, height=400)

    csv = top_rules_preview.to_csv(index=False).encode('utf-8')

    #logs download event
    def log_download():
        log_event(action_type="download", page_name="Market Basket Intelligence", file_name="market_basket_top500_rules.csv")

    st.download_button(label="Download Rules", data=csv, file_name="market_basket_top500_rules.csv",
                       mime="text/csv", on_click=log_download)