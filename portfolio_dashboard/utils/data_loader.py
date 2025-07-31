
"""
Módulo para carga y conexión de datos del portafolio.
"""

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
from typing import Tuple, Dict, Optional

# Configuración de credenciales
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

def get_credentials() -> Dict:
    """Obtener credenciales de Google Cloud según el entorno."""
    try:
        # Desarrollo local - usar secrets.toml
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            return st.secrets["gcp_service_account"]
        
        # Producción - usar variables de entorno
        elif os.getenv('GCP_SERVICE_ACCOUNT'):
            return json.loads(os.getenv('GCP_SERVICE_ACCOUNT'))
        
        else:
            st.error("❌ No se encontraron credenciales de Google Cloud configuradas")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error al obtener credenciales: {str(e)}")
        st.stop()

def get_sheet_id() -> str:
    """Obtener sheet_id según el entorno."""
    try:
        # Desarrollo local
        if hasattr(st, 'secrets') and 'sheet_id' in st.secrets:
            return st.secrets["sheet_id"]
        
        # Producción
        elif os.getenv('SHEET_ID'):
            return os.getenv('SHEET_ID')
        
        else:
            st.error("❌ No se encontró sheet_id configurado")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error al obtener sheet_id: {str(e)}")
        st.stop()

def conectar_google_sheets():
    """Establecer conexión con Google Sheets."""
    try:
        credentials_dict = get_credentials()
        
        # Configurar scopes
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Crear credenciales y autorizar cliente
        credentials = Credentials.from_service_account_info(
            credentials_dict, 
            scopes=scopes
        )
        
        gc = gspread.authorize(credentials)
        return gc

    except Exception as e:
        st.error(f"❌ Error al conectar con Google Sheets: {str(e)}")
        st.stop()

@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_sheet_data(worksheet_name: str) -> pd.DataFrame:
    """Cargar datos de un worksheet específico."""
    try:
        gc = conectar_google_sheets()
        sheet_id = get_sheet_id()
        
        # Abrir el sheet y obtener worksheet
        sheet = gc.open_by_key(sheet_id)
        worksheet = sheet.worksheet(worksheet_name)
        
        # Obtener datos y convertir a DataFrame
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Limpiar datos vacíos
        df = df.replace('', pd.NA).dropna(how='all')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error al cargar {worksheet_name}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_portfolio_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cargar todos los datos del portafolio."""
    
    with st.spinner("Cargando datos del portafolio..."):
        # Cargar todos los sheets
        operaciones = load_sheet_data("Operaciones")
        transacciones = load_sheet_data("Transacciones")
        dividendos = load_sheet_data("Dividendos")
        history = load_sheet_data("Daily_values")
        
        # Procesar fechas
        operaciones = process_dates(operaciones, 'Fecha')
        transacciones = process_dates(transacciones, 'Fecha')
        dividendos = process_dates(dividendos, 'Fecha')
        history = process_dates(history, "Fecha")
        
        # Limpiar y validar datos
        operaciones = clean_operaciones_data(operaciones)
        transacciones = clean_transacciones_data(transacciones)
        dividendos = clean_dividendos_data(dividendos)
    
    return operaciones, transacciones, dividendos, history

def process_dates(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Procesar y convertir columnas de fecha."""
    if df.empty or date_column not in df.columns:
        return df
    
    try:
        # Convertir a datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Eliminar filas con fechas inválidas
        df = df.dropna(subset=[date_column])
        
        # Ordenar por fecha
        df = df.sort_values(date_column).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Error al procesar fechas en {date_column}: {str(e)}")
        return df

def clean_operaciones_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar y validar datos de operaciones."""
    if df.empty:
        return df
    
    try:
        # Convertir columnas numéricas
        numeric_columns = ['Cantidad', 'Precio', 'Valor', 'Comision', 'Impuestos']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Limpiar símbolos (mayúsculas, sin espacios)
        if 'Simbolo' in df.columns:
            df['Simbolo'] = df['Simbolo'].astype(str).str.strip().str.upper()
        
        # Validar tipos de operación
        if 'Tipo' in df.columns:
            valid_types = ['Compra', 'Venta', 'COMPRA', 'VENTA']
            df = df[df['Tipo'].isin(valid_types)]
            df['Tipo'] = df['Tipo'].str.title()  # Capitalizar
        
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Error al limpiar datos de operaciones: {str(e)}")
        return df

def clean_transacciones_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar y validar datos de transacciones."""
    if df.empty:
        return df
    
    try:
        # Convertir monto a numérico
        if 'Monto' in df.columns:
            df['Monto'] = pd.to_numeric(df['Monto'], errors='coerce')
        
        # Validar tipos de transacción
        if 'Tipo' in df.columns:
            valid_types = ['Deposito', 'Retiro', 'DEPOSITO', 'RETIRO', 'Depósito']
            df = df[df['Tipo'].isin(valid_types)]
            df['Tipo'] = df['Tipo'].str.replace('ó', 'o').str.title()  # Normalizar
        
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Error al limpiar datos de transacciones: {str(e)}")
        return df

def clean_dividendos_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpiar y validar datos de dividendos."""
    if df.empty:
        return df
    
    try:
        # Convertir columnas numéricas
        numeric_columns = ['Dividendo_por_Accion', 'Cantidad_Acciones', 'Total', 'Impuestos']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Limpiar símbolos
        if 'Simbolo' in df.columns:
            df['Simbolo'] = df['Simbolo'].astype(str).str.strip().str.upper()
        
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Error al limpiar datos de dividendos: {str(e)}")
        return df

@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_benchmark_data(symbol: str, start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
    """Obtener datos de benchmarks (S&P 500, etc.) desde Yahoo Finance."""
    
    if end_date is None:
        end_date = datetime.now()
    
    try:
        with st.spinner(f"Descargando datos de {symbol}..."):
            # Descargar datos
            data = yf.download(
                symbol, 
                start=start_date, 
                end=end_date,
                progress=False
            )
            
            if data.empty:
                st.warning(f"⚠️ No se encontraron datos para {symbol}")
                return pd.DataFrame()
            
            # Resetear índice para tener fecha como columna
            data = data.reset_index()
            
            # Renombrar columnas
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            data['Rendimiento'] = (data['Close']/data.loc[0, 'Close'] - 1) * 100
            
            return data
            
    except Exception as e:
        st.error(f"❌ Error al obtener datos de {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_current_prices(symbols: list, date: datetime = None) -> Dict[str, float]:
    """Obtener precios actuales de una lista de símbolos."""
    
    if not symbols:
        return {}
    
    try:
        with st.spinner("Obteniendo precios actuales..."):
            prices = {}

            if not date:
                date = datetime.now()
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    history = ticker.history(start=date, end=date+timedelta(days=5))
                    info = ticker.info
                    
                    # Intentar diferentes campos de precio
                    if not history.empty and len(history) > 0:
                        price = history['Close'].iat[0]
                    else:
                        price = info['previousClose']
                    
                    if price:
                        prices[symbol] = float(price)
                    else:
                        # Fallback: obtener último precio de datos históricos
                        hist = ticker.history(period="1d")
                        if not hist.empty:
                            prices[symbol] = float(hist['Close'].loc[history.index.asof(date)])
                            
                except Exception as e:
                    st.warning(f"⚠️ No se pudo obtener precio para {symbol}: {str(e)}")
                    continue
            
            return prices
            
    except Exception as e:
        st.error(f"❌ Error al obtener precios actuales: {str(e)}")
        return {}

def get_available_symbols(operaciones: pd.DataFrame) -> list:
    """Obtener lista de símbolos únicos del portafolio."""
    if operaciones.empty or 'Simbolo' not in operaciones.columns:
        return []
    
    return sorted(operaciones['Simbolo'].dropna().unique().tolist())

def validate_data_integrity(operaciones: pd.DataFrame, transacciones: pd.DataFrame, dividendos: pd.DataFrame) -> Dict[str, bool]:
    """Validar la integridad de los datos cargados."""
    
    validation_results = {
        'operaciones_valid': True,
        'transacciones_valid': True,
        'dividendos_valid': True,
        'dates_consistent': True
    }
    
    # Validar operaciones
    if not operaciones.empty:
        required_cols = ['Fecha', 'Simbolo', 'Tipo', 'Cantidad', 'Precio']
        missing_cols = [col for col in required_cols if col not in operaciones.columns]
        if missing_cols:
            st.warning(f"⚠️ Columnas faltantes en Operaciones: {missing_cols}")
            validation_results['operaciones_valid'] = False
    
    # Validar transacciones
    if not transacciones.empty:
        required_cols = ['Fecha', 'Tipo', 'Monto']
        missing_cols = [col for col in required_cols if col not in transacciones.columns]
        if missing_cols:
            st.warning(f"⚠️ Columnas faltantes en Transacciones: {missing_cols}")
            validation_results['transacciones_valid'] = False
    
    # Validar dividendos
    if not dividendos.empty:
        required_cols = ['Fecha', 'Simbolo', 'Total']
        missing_cols = [col for col in required_cols if col not in dividendos.columns]
        if missing_cols:
            st.warning(f"⚠️ Columnas faltantes en Dividendos: {missing_cols}")
            validation_results['dividendos_valid'] = False
    
    return validation_results