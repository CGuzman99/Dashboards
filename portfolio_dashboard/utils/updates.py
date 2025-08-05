import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
from typing import Tuple, Dict, Optional

import utils.data_loader as data_loader
from utils.calculations import calculate_daily_portfolio_values

def write_sheet_data(worksheet_name: str, data: pd.DataFrame):
    """ Escribir dataframe en worksheet """
    try:
        gc = data_loader.conectar_google_sheets()
        sheet_id = data_loader.get_sheet_id()
        sheet = gc.open_by_key(sheet_id)
        worksheet = sheet.worksheet(worksheet_name)

        if not data.empty:
            data_to_write = data.copy()
            data_to_write['Fecha'] = data_to_write["Fecha"].astype(str)

            worksheet.clear()
            worksheet.update([data_to_write.columns.values.tolist()] + data_to_write.values.tolist())

    except Exception as e:
        st.error(f"Error escribiendo en {worksheet_name}: {str(e)}")
        raise

@st.cache_data(ttl=1800)  # Cache por 30 minutos
def check_last_update_time():
    """Verifica cuándo fue la última actualización"""
    return datetime.now()

def update_portfolio_history():
    """ Actualiza los valores diarios del portafolio en el archivo """

    # Evitar actualizaciones muy frecuentes
    last_check = check_last_update_time()
    
    with st.spinner("Verificando actualizaciones del portafolio..."):

        operaciones, transacciones, dividendos, history = data_loader.load_portfolio_data()

        if not history.empty:
            last_date = pd.to_datetime(history['Fecha'].iloc[-1])
        else:
            last_date = pd.to_datetime('2025-03-23')
        today = datetime.now()

        # Solo actualizar si faltan datos de días hábiles
        dias_faltantes = pd.bdate_range(start=last_date + timedelta(days=1), end=today)

        if len(dias_faltantes) == 0:
            #st.success("✅ Datos del portafolio actualizados")
            return False

        try:
            new_values = calculate_daily_portfolio_values(
                operaciones, transacciones, dividendos, 
                start_date=last_date + timedelta(days=1)
            )

            if not new_values.empty:
                new_values['Fecha'] = pd.to_datetime(new_values['Fecha'], errors='coerce')
                new_history = pd.concat([history, new_values], ignore_index=True)

                data_loader.load_portfolio_data.clear()

                write_sheet_data("Daily_values", new_history)
                st.success(f"✅ Actualizados {len(new_values)} días de datos")
                return True
            else:
                st.info("ℹ️ No hay datos que actualizar")
                return False

        except Exception as e:
            st.error(f"❌ Error actualizando portafolio: {str(e)}")
            return False