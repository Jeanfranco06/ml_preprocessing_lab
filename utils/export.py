# Módulo de exportación básica
# Funciones simples para exportar datos

import pandas as pd
import streamlit as st

def export_to_csv(df, filename):
    """Exportar dataframe a CSV"""
    csv_data = df.to_csv(index=False)
    st.download_button(
        f"📥 Descargar {filename}.csv",
        csv_data,
        file_name=f"{filename}.csv",
        mime="text/csv"
    )
