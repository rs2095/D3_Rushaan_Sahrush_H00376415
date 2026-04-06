import streamlit as st


#injects custom css styles for all dashboard pages
def inject_styles():
    st.markdown("""
    <style>
    .card, .sust-card, .market-card, .sim-card {
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    body, div, p, h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif; }

    .css-1d391kg { background-color: #0e1117; }

    .card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left-width: 4px;
        border-left-style: solid;
        padding: 1rem;
        border-radius: 0.75rem;
        background-color: #1e222e;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(106, 13, 173, 0.15);
    }

    .border-purple-600 { border-left-color: #7B3FE4; }
    .border-purple-500 { border-left-color: #8a52ff; }
    .border-purple-400 { border-left-color: #9b6fff; }
    .border-purple-300 { border-left-color: #aa88ff; }

    .text-gray-500 { color: #b0b3c0; }
    .text-gray-600 { color: #c0c3d0; }
    .text-gray-700 { color: #d0d3e0; }
    .text-gray-800 { color: #e0e3f0; }
    .text-green-600 { color: #00ff99; }
    .text-orange-600 { color: #ff9f43; }

    .kpi-title { font-size: 0.75rem; font-weight: 500; text-transform: uppercase; color: #b0b3c0; }
    .kpi-value { font-size: 1.75rem; font-weight: 700; margin-top: 0.25rem; color: #e0e3f0; }
    .kpi-sub { font-size: 0.75rem; margin-top: 0.25rem; display: flex; align-items: center; gap: 0.25rem; }

    .sust-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left-width: 4px;
        border-left-style: solid;
        padding: 1rem;
        border-radius: 0.75rem;
        background-color: #1e222e;
    }
    .sust-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(46, 125, 50, 0.15);
    }
    .sust-border-green-600 { border-left-color: #2E7D32; }
    .sust-border-green-500 { border-left-color: #66BB6A; }
    .sust-kpi-title { font-size: 0.75rem; font-weight: 500; text-transform: uppercase; color: #b0b3c0; }
    .sust-kpi-value { font-size: 1.75rem; font-weight: 700; margin-top: 0.25rem; color: #e0e3f0; }
    .sust-kpi-sub { font-size: 0.75rem; margin-top: 0.25rem; display: flex; align-items: center; gap: 0.25rem; }

    .market-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left-width: 4px;
        border-left-style: solid;
        padding: 1rem;
        border-radius: 0.75rem;
        background-color: #1e222e;
    }
    .market-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(79, 195, 247, 0.15);
    }

    .border-blue-600 { border-left-color: #29B6F6; }
    .border-blue-500 { border-left-color: #4FC3F7; }
    .border-blue-400 { border-left-color: #81D4FA; }
    .border-blue-300 { border-left-color: #B3E5FC; }

    .text-blue-600 { color: #29B6F6; }
    .text-blue-500 { color: #4FC3F7; }

    .market-kpi-title { font-size: 0.75rem; font-weight: 500; text-transform: uppercase; color: #b0b3c0; }
    .market-kpi-value { font-size: 1.75rem; font-weight: 700; margin-top: 0.25rem; color: #e0e3f0; }
    .market-kpi-sub { font-size: 0.75rem; margin-top: 0.25rem; display: flex; align-items: center; gap: 0.25rem; }

    .sim-card {
        min-height: unset;
        height: fit-content;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left-width: 4px;
        border-left-style: solid;
        padding: 1rem;
        border-radius: 0.75rem;
        background-color: #1e222e;
    }
    .sim-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(255, 255, 255, 0.05);
    }

    .sim-title {
        font-size: 14px;
        color: #9ca3af;
        margin-bottom: 4px;
    }
    .sim-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .sim-sub {
        font-size: 13px;
        color: #d1d5db;
    }

    .sim-header {
        border-left: 4px solid #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        background-color: #1e222e;
    }

    </style>
    """, unsafe_allow_html=True)