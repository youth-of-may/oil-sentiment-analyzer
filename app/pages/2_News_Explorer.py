import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Geopolitical Sentiment & Oil Price Correlation", page_icon="⛽", layout="wide")

