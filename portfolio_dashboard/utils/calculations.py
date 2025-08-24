"""
Módulo para cálculos financieros del portafolio.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import streamlit as st

from utils.data_loader import get_available_symbols, get_current_prices

def calculate_current_holdings(operaciones: pd.DataFrame) -> pd.DataFrame:
    """Calcular posiciones actuales basadas en operaciones."""
    if operaciones.empty:
        return pd.DataFrame()
    
    try:
        # Agrupar por símbolo y calcular posición neta
        holdings = []
        
        for simbolo in operaciones['Simbolo'].unique():
            ops_simbolo = operaciones[operaciones['Simbolo'] == simbolo].copy()
            
            # Calcular cantidad neta
            compras = ops_simbolo[ops_simbolo['Tipo'] == 'Compra']['Cantidad'].sum()
            ventas = ops_simbolo[ops_simbolo['Tipo'] == 'Venta']['Cantidad'].sum()
            cantidad_neta = compras - ventas
            
            if cantidad_neta > 0:  # Solo incluir posiciones positivas
                # Calcular precio promedio ponderado (solo compras)
                compras_data = ops_simbolo[ops_simbolo['Tipo'] == 'Compra']
                if not compras_data.empty:
                    precio_promedio = (compras_data['Cantidad'] * compras_data['Precio']).sum() / compras_data['Cantidad'].sum()
                    
                    # Calcular costo total (incluyendo comisiones e impuestos)
                    costo_total = compras_data['Valor'].sum() + compras_data.get('Comision', 0).sum() + compras_data.get('Impuestos', 0).sum()

                    holdings.append({
                        'Simbolo': simbolo,
                        'Cantidad': cantidad_neta,
                        'Precio_Promedio': precio_promedio,
                        'Costo_Total': costo_total,
                        'Fecha_Primera_Compra': compras_data['Fecha'].min(),
                        'Fecha_Ultima_Operacion': ops_simbolo['Fecha'].max()
                    })
        
        return pd.DataFrame(holdings)
        
    except Exception as e:
        st.error(f"❌ Error al calcular holdings actuales: {str(e)}")
        return pd.DataFrame()

def calculate_portfolio_value(holdings: pd.DataFrame, current_prices: Dict[str, float]) -> Dict:
    """Calcular valor actual del portafolio."""
    if holdings.empty or not current_prices:
        return {
            'valor_total': 0,
            'costo_total': 0,
            'ganancia_perdida': 0,
            'rendimiento_porcentaje': 0
        }
    
    try:
        valor_total = 0
        costo_total = 0
        
        for _, holding in holdings.iterrows():
            simbolo = holding['Simbolo']
            cantidad = holding['Cantidad']
            costo = holding['Costo_Total']
            
            if simbolo in current_prices:
                precio_actual = current_prices[simbolo]
                valor_actual = cantidad * precio_actual
                valor_total += valor_actual
            
            costo_total += costo
        
        ganancia_perdida = valor_total - costo_total
        rendimiento_porcentaje = (ganancia_perdida / costo_total * 100) if costo_total > 0 else 0
        
        return {
            'valor_total': valor_total,
            'costo_total': costo_total,
            'ganancia_perdida': ganancia_perdida,
            'rendimiento_porcentaje': rendimiento_porcentaje
        }
        
    except Exception as e:
        st.error(f"❌ Error al calcular valor del portafolio: {str(e)}")
        return {
            'valor_total': 0,
            'costo_total': 0,
            'ganancia_perdida': 0,
            'rendimiento_porcentaje': 0
        }

def calculate_cash_position(transacciones: pd.DataFrame, operaciones: pd.DataFrame, dividendos: pd.DataFrame) -> float:
    """Calcular posición de efectivo actual."""
    try:
        # Depósitos y retiros
        depositos = transacciones[transacciones['Tipo'] == 'Deposito']['Monto'].sum() if not transacciones.empty else 0
        retiros = transacciones[transacciones['Tipo'] == 'Retiro']['Monto'].sum() if not transacciones.empty else 0
        
        # Operaciones de compra y venta
        compras = operaciones[operaciones['Tipo'] == 'Compra']['Valor'].sum() if not operaciones.empty else 0
        ventas = operaciones[operaciones['Tipo'] == 'Venta']['Valor'].sum() if not operaciones.empty else 0
        comisiones = operaciones['Comision'].sum() if not operaciones.empty and 'Comision' in operaciones.columns else 0
        impuestos_operaciones = operaciones['Impuestos'].sum() if not operaciones.empty and 'Impuestos' in operaciones.columns else 0
        
        # Dividendos recibidos
        dividendos_total = dividendos['Total'].sum() if not dividendos.empty else 0
        dividendos_impuestos = dividendos['Impuestos'].sum() if not dividendos.empty else 0
        
        # Cálculo final
        efectivo = depositos - retiros - compras + ventas - comisiones - impuestos_operaciones + dividendos_total - dividendos_impuestos
        
        return efectivo #max(efectivo, 0)  # No permitir efectivo negativo
        
    except Exception as e:
        st.error(f"❌ Error al calcular posición de efectivo: {str(e)}")
        return 0

def calculate_daily_portfolio_values(operaciones: pd.DataFrame, transacciones: pd.DataFrame, 
                                   dividendos: pd.DataFrame, start_date: datetime = None) -> pd.DataFrame:
    """Calcular valores diarios del portafolio."""
    if operaciones.empty and transacciones.empty:
        return pd.DataFrame()
    
    try:
        # Determinar rango de fechas
        all_dates = []
        if not transacciones.empty:
            all_dates.extend(transacciones['Fecha'].tolist())
        if not operaciones.empty:
            all_dates.extend(operaciones['Fecha'].tolist())       
        if not dividendos.empty:
            all_dates.extend(dividendos['Fecha'].tolist())
        
        if not all_dates:
            return pd.DataFrame()
        
        fecha_inicio = start_date or min(all_dates)
        fecha_fin = datetime.now()
        
        # Crear rango de fechas diarias
        date_range = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        
        # Por simplicidad, calculamos valores semanales para mejor rendimiento
        weekly_dates = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='W')
        weekly_dates = weekly_dates + timedelta(days=1)
        
        portfolio_values = []

        # Para TWR
        twr_acumulado = 1.0
        valor_inicial_subperiodo = None
        
        for date in date_range:
            # Omitir fin de semana
            if date.weekday() > 4:
                continue
            # Operaciones hasta esta fecha
            ops_hasta_fecha = operaciones[operaciones['Fecha'] <= date] if not operaciones.empty else pd.DataFrame()
            trans_hasta_fecha = transacciones[transacciones['Fecha'] <= date] if not transacciones.empty else pd.DataFrame()
            div_hasta_fecha = dividendos[dividendos['Fecha'] <= date] if not dividendos.empty else pd.DataFrame()
            
            # Calcular holdings en esta fecha
            holdings = calculate_current_holdings(ops_hasta_fecha)
            symbols = get_available_symbols(holdings)
            current_prices = get_current_prices(symbols, date)
            valor_posiciones = sum(
                holdings.at[row, 'Cantidad'] * current_prices[holdings.at[row, 'Simbolo']]
                for row in holdings.index
            )

            # Calcular efectivo
            efectivo = calculate_cash_position(trans_hasta_fecha, ops_hasta_fecha, div_hasta_fecha)
            valor_portafolio = valor_posiciones + efectivo
            
            # Detectar si es el inicio del subperiodo
            if valor_inicial_subperiodo is None:
                valor_inicial_subperiodo = valor_portafolio
            
            r_sub = (valor_portafolio / valor_inicial_subperiodo) - 1
            twr_acumulado *= (1 + r_sub)

            # Si hay flujo de caja en esta fecha -> cerramos subperiodo
            trans_hoy = transacciones[transacciones['Fecha'] == date] if not transacciones.empty else pd.DataFrame()
            if not trans_hoy.empty:
                valor_inicial_subperiodo = valor_portafolio  # reinicia subperiodo
            
            
            # Rendimiento acumulado TWR
            rendimiento_twr = (twr_acumulado - 1) * 100
            
            # Calcular valor total (simplificado sin precios históricos)
            costo_invertido = ops_hasta_fecha[ops_hasta_fecha['Tipo'] == 'Compra']['Valor'].sum() if not ops_hasta_fecha.empty else 0
            
            #valor_posiciones = sum(positions_values.values())
            rendimiento = ((valor_posiciones + efectivo)/(costo_invertido + efectivo) - 1) * 100
            
            portfolio_values.append({
                'Fecha': date,
                'Efectivo': efectivo,
                #'Costo_Invertido': costo_invertido,
                'Valor_Posiciones': valor_posiciones,
                'Valor_Portafolio': valor_portafolio,
                'Rendimiento': rendimiento,
                'Num_Posiciones': len(holdings),
                'Dividendos_Acumulados': div_hasta_fecha['Total'].sum() if not div_hasta_fecha.empty else 0
            })
        
        return pd.DataFrame(portfolio_values)
        
    except Exception as e:
        st.error(f"❌ Error al calcular valores diarios: {str(e)}")
        return pd.DataFrame()

def calculate_returns(portfolio_values: pd.DataFrame, benchmark_data: pd.DataFrame = None) -> Dict:
    """Calcular métricas de rendimiento."""
    if portfolio_values.empty:
        return {}
    
    try:
        # Calcular rendimientos del portafolio
        portfolio_values = portfolio_values.sort_values('Fecha')
        
        # Rendimiento simple
        valor_inicial = portfolio_values['Costo_Invertido'].iloc[0] if len(portfolio_values) > 0 else 0
        valor_final = portfolio_values['Costo_Invertido'].iloc[-1] if len(portfolio_values) > 0 else 0
        
        rendimiento_total = ((valor_final - valor_inicial) / valor_inicial * 100) if valor_inicial > 0 else 0
        
        # Calcular período en días
        fecha_inicial = portfolio_values['Fecha'].iloc[0]
        fecha_final = portfolio_values['Fecha'].iloc[-1]
        dias_total = (fecha_final - fecha_inicial).days
        
        # Rendimiento anualizado
        if dias_total > 0:
            rendimiento_anualizado = ((valor_final / valor_inicial) ** (365 / dias_total) - 1) * 100 if valor_inicial > 0 else 0
        else:
            rendimiento_anualizado = 0
        
        results = {
            'rendimiento_total': rendimiento_total,
            'rendimiento_anualizado': rendimiento_anualizado,
            'dias_inversion': dias_total,
            'valor_inicial': valor_inicial,
            'valor_final': valor_final
        }
        
        # Agregar comparación con benchmark si está disponible
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_return = calculate_benchmark_return(benchmark_data, fecha_inicial, fecha_final)
            results['benchmark_rendimiento'] = benchmark_return
            results['alpha'] = rendimiento_anualizado - benchmark_return
        
        return results
        
    except Exception as e:
        st.error(f"❌ Error al calcular rendimientos: {str(e)}")
        return {}

def calculate_benchmark_return(benchmark_data: pd.DataFrame, start_date: datetime, end_date: datetime) -> float:
    """Calcular rendimiento de benchmark en período específico."""
    try:
        # Filtrar datos por fechas
        mask = (benchmark_data['Date'] >= start_date) & (benchmark_data['Date'] <= end_date)
        period_data = benchmark_data.loc[mask]
        
        if len(period_data) < 2:
            return 0
        
        precio_inicial = period_data['Close'].iloc[0]
        precio_final = period_data['Close'].iloc[-1]
        
        # Calcular rendimiento anualizado
        dias = (end_date - start_date).days
        if dias > 0 and precio_inicial > 0:
            rendimiento = ((precio_final / precio_inicial) ** (365 / dias) - 1) * 100
            return rendimiento
        
        return 0
        
    except Exception as e:
        st.warning(f"⚠️ Error al calcular rendimiento de benchmark: {str(e)}")
        return 0

def calculate_volatility(returns_series: pd.Series) -> float:
    """Calcular volatilidad (desviación estándar) de una serie de rendimientos."""
    try:
        if returns_series.empty or len(returns_series) < 2:
            return 0
        
        # Calcular rendimientos diarios
        daily_returns = returns_series.pct_change().dropna()
        
        # Volatilidad anualizada
        volatility = daily_returns.std() * np.sqrt(252) * 100  # 252 días de trading por año
        
        return volatility if not np.isnan(volatility) else 0
        
    except Exception as e:
        st.warning(f"⚠️ Error al calcular volatilidad: {str(e)}")
        return 0

def calculate_sharpe_ratio(portfolio_return: float, volatility: float, risk_free_rate: float = 2.0) -> float:
    """Calcular ratio de Sharpe."""
    try:
        if volatility == 0:
            return 0
        
        excess_return = portfolio_return - risk_free_rate
        sharpe = excess_return / volatility
        
        return sharpe if not np.isnan(sharpe) else 0
        
    except Exception as e:
        st.warning(f"⚠️ Error al calcular Sharpe ratio: {str(e)}")
        return 0

def calculate_max_drawdown(values_series: pd.Series) -> Dict:
    """Calcular drawdown máximo."""
    try:
        if values_series.empty:
            return {'max_drawdown': 0, 'drawdown_duration': 0}
        
        # Calcular peak running (máximo histórico)
        peak = values_series.expanding().max()
        
        # Calcular drawdown
        drawdown = (values_series - peak) / peak * 100
        
        # Encontrar drawdown máximo
        max_drawdown = drawdown.min()
        
        # Calcular duración del drawdown máximo
        max_dd_date = drawdown.idxmin()
        recovery_date = values_series[values_series.index > max_dd_date].loc[
            values_series[values_series.index > max_dd_date] >= peak.loc[max_dd_date]
        ].index
        
        if len(recovery_date) > 0:
            duration = (recovery_date[0] - max_dd_date).days
        else:
            duration = (values_series.index[-1] - max_dd_date).days
        
        return {
            'max_drawdown': abs(max_drawdown) if not np.isnan(max_drawdown) else 0,
            'drawdown_duration': duration
        }
        
    except Exception as e:
        st.warning(f"⚠️ Error al calcular drawdown máximo: {str(e)}")
        return {'max_drawdown': 0, 'drawdown_duration': 0}

def calculate_portfolio_allocation(holdings: pd.DataFrame, current_prices: Dict[str, float]) -> pd.DataFrame:
    """Calcular asignación del portafolio por instrumento."""
    if holdings.empty or not current_prices:
        return pd.DataFrame()
    
    try:
        allocation_data = []
        total_value = 0
        
        # Calcular valor actual de cada posición
        for _, holding in holdings.iterrows():
            simbolo = holding['Simbolo']
            cantidad = holding['Cantidad']
            
            if simbolo in current_prices:
                precio_actual = current_prices[simbolo]
                valor_actual = cantidad * precio_actual
                total_value += valor_actual
                
                allocation_data.append({
                    'Simbolo': simbolo,
                    'Cantidad': cantidad,
                    'Precio_Actual': precio_actual,
                    'Valor_Actual': valor_actual,
                    'Costo_Total': holding['Costo_Total'],
                    'Ganancia_Perdida': valor_actual - holding['Costo_Total'],
                    'Rendimiento_Pct': ((valor_actual - holding['Costo_Total']) / holding['Costo_Total'] * 100) if holding['Costo_Total'] > 0 else 0
                })
        
        allocation_df = pd.DataFrame(allocation_data)
        
        # Calcular porcentajes de asignación
        if not allocation_df.empty and total_value > 0:
            allocation_df['Peso_Porcentaje'] = (allocation_df['Valor_Actual'] / total_value * 100).round(2)
        
        return allocation_df.sort_values('Valor_Actual', ascending=False)
        
    except Exception as e:
        st.error(f"❌ Error al calcular asignación del portafolio: {str(e)}")
        return pd.DataFrame()

def calculate_dividend_yield(dividendos: pd.DataFrame, holdings: pd.DataFrame, current_prices: Dict[str, float]) -> Dict:
    """Calcular rendimiento por dividendos."""
    if dividendos.empty or holdings.empty:
        return {'dividend_yield': 0, 'dividendos_anuales': 0}
    
    try:
        # Dividendos del último año
        fecha_hace_un_año = datetime.now() - timedelta(days=365)
        dividendos_recientes = dividendos[dividendos['Fecha'] >= fecha_hace_un_año]
        
        dividendos_anuales = dividendos_recientes['Total'].sum()
        
        # Valor actual del portafolio
        valor_portafolio = sum(
            holding['Cantidad'] * current_prices.get(holding['Simbolo'], 0)
            for _, holding in holdings.iterrows()
            if holding['Simbolo'] in current_prices
        )
        
        # Calcular dividend yield
        dividend_yield = (dividendos_anuales / valor_portafolio * 100) if valor_portafolio > 0 else 0
        
        return {
            'dividend_yield': dividend_yield,
            'dividendos_anuales': dividendos_anuales,
            'valor_portafolio': valor_portafolio
        }
        
    except Exception as e:
        st.warning(f"⚠️ Error al calcular dividend yield: {str(e)}")
        return {'dividend_yield': 0, 'dividendos_anuales': 0}

def calculate_sector_allocation(holdings: pd.DataFrame, current_prices: Dict[str, float]) -> pd.DataFrame:
    """Calcular asignación por sector (simplificado por símbolo)."""
    if holdings.empty:
        return pd.DataFrame()
    
    try:
        # Mapeo básico de sectores (esto se puede expandir)
        sector_mapping = {
            'NVDA.MX': 'Technology',
            'AMD.MX': 'Technology',
            'MSFT': 'Technology', 
            'GOOG.MX': 'Communication Services',
            'AMZN.MX': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'META': 'Technology',
            'NVDA': 'Technology',
            'UBER.MX': 'Technology',
            'BABAN.MX': 'Consumer Cyclical',
            'CRSPN.MX': 'Healthcare',
            'IVVPESOISHRS.MX': 'ETF - Diversified',
            'QQQM.MX': 'ETF - Technology',
            'VTI': 'ETF - Total Market',
            'VOO': 'ETF - S&P 500'
        }
        
        sector_data = {}
        total_value = 0
        
        for _, holding in holdings.iterrows():
            simbolo = holding['Simbolo']
            cantidad = holding['Cantidad']
            
            if simbolo in current_prices:
                precio_actual = current_prices[simbolo]
                valor_actual = cantidad * precio_actual
                total_value += valor_actual
                
                # Determinar sector
                sector = sector_mapping.get(simbolo, 'Other')
                
                if sector in sector_data:
                    sector_data[sector] += valor_actual
                else:
                    sector_data[sector] = valor_actual
        
        # Convertir a DataFrame
        sector_df = pd.DataFrame([
            {'Sector': sector, 'Valor': valor, 'Porcentaje': (valor / total_value * 100) if total_value > 0 else 0}
            for sector, valor in sector_data.items()
        ])
        
        return sector_df.sort_values('Valor', ascending=False)
        
    except Exception as e:
        st.warning(f"⚠️ Error al calcular asignación por sector: {str(e)}")
        return pd.DataFrame()

def calculate_metrics_summary(operaciones: pd.DataFrame, transacciones: pd.DataFrame, 
                            dividendos: pd.DataFrame, current_prices: Dict[str, float]) -> Dict:
    """Calcular resumen completo de métricas del portafolio."""
    try:
        # Calcular holdings actuales
        holdings = calculate_current_holdings(operaciones)
        
        # Valor del portafolio
        portfolio_value = calculate_portfolio_value(holdings, current_prices)
        
        # Posición de efectivo
        efectivo = calculate_cash_position(transacciones, operaciones, dividendos)
        
        # Asignación del portafolio
        allocation = calculate_portfolio_allocation(holdings, current_prices)
        
        # Dividend yield
        dividend_metrics = calculate_dividend_yield(dividendos, holdings, current_prices)
        
        # Valores diarios para cálculos de volatilidad
        daily_values = calculate_daily_portfolio_values(operaciones, transacciones, dividendos)
        
        # Rendimientos
        returns_metrics = calculate_returns(daily_values)
        
        # Consolidar todas las métricas
        summary = {
            # Valores principales
            'valor_total': portfolio_value['valor_total'] + efectivo,
            'valor_inversiones': portfolio_value['valor_total'],
            'efectivo': efectivo,
            'costo_total': portfolio_value['costo_total'],
            
            # Rendimiento
            'ganancia_perdida_total': portfolio_value['ganancia_perdida'],
            'rendimiento_porcentaje': portfolio_value['rendimiento_porcentaje'],
            'rendimiento_anualizado': returns_metrics.get('rendimiento_anualizado', 0),
            
            # Dividendos
            'dividend_yield': dividend_metrics['dividend_yield'],
            'dividendos_anuales': dividend_metrics['dividendos_anuales'],
            
            # Posiciones
            'num_posiciones': len(holdings),
            'simbolos_activos': holdings['Simbolo'].tolist() if not holdings.empty else [],
            
            # Estadísticas adicionales
            'dias_inversion': returns_metrics.get('dias_inversion', 0),
            'primera_inversion': operaciones['Fecha'].min() if not operaciones.empty else None,
            'ultima_operacion': operaciones['Fecha'].max() if not operaciones.empty else None,
            
            # Total de transacciones
            'total_depositos': transacciones[transacciones['Tipo'] == 'Deposito']['Monto'].sum() if not transacciones.empty else 0,
            'total_retiros': transacciones[transacciones['Tipo'] == 'Retiro']['Monto'].sum() if not transacciones.empty else 0,
            'total_compras': operaciones[operaciones['Tipo'] == 'Compra']['Valor'].sum() if not operaciones.empty else 0,
            'total_ventas': operaciones[operaciones['Tipo'] == 'Venta']['Valor'].sum() if not operaciones.empty else 0,
            'total_comisiones': operaciones['Comision'].sum() if not operaciones.empty and 'Comision' in operaciones.columns else 0,
            'total_impuestos_operaciones': operaciones['Impuestos'].sum() if not operaciones.empty and 'Impuestos' in operaciones.columns else 0,
            'total_impuestos_dividendos': dividendos['Impuestos'].sum() if not dividendos.empty and 'Impuestos' in dividendos.columns else 0,
        }
        
        return summary
        
    except Exception as e:
        st.error(f"❌ Error al calcular resumen de métricas: {str(e)}")
        return {}


def calculate_monthly_performance(operaciones: pd.DataFrame, transacciones: pd.DataFrame, 
                                dividendos: pd.DataFrame, current_prices: Dict[str, float]) -> pd.DataFrame:
    """Calcular rendimiento mensual del portafolio."""
    try:
        if operaciones.empty:
            return pd.DataFrame()
        
        # Obtener rango de fechas
        fecha_inicio = operaciones['Fecha'].min()
        fecha_fin = datetime.now()
        
        # Crear fechas mensuales
        monthly_dates = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='M')
        
        monthly_performance = []
        
        for i, date in enumerate(monthly_dates):
            # Datos hasta esta fecha
            ops_hasta_fecha = operaciones[operaciones['Fecha'] <= date]
            trans_hasta_fecha = transacciones[transacciones['Fecha'] <= date] if not transacciones.empty else pd.DataFrame()
            div_hasta_fecha = dividendos[dividendos['Fecha'] <= date] if not dividendos.empty else pd.DataFrame()
            
            # Calcular métricas del mes
            holdings = calculate_current_holdings(ops_hasta_fecha)
            efectivo = calculate_cash_position(trans_hasta_fecha, ops_hasta_fecha, div_hasta_fecha)
            
            # Valor total invertido
            costo_total = ops_hasta_fecha[ops_hasta_fecha['Tipo'] == 'Compra']['Valor'].sum()
            
            monthly_performance.append({
                'Fecha': date.strftime('%Y-%m'),
                'Mes': date.strftime('%B %Y'),
                'Costo_Acumulado': costo_total,
                'Efectivo': efectivo,
                'Num_Posiciones': len(holdings),
                'Dividendos_Mes': div_hasta_fecha[div_hasta_fecha['Fecha'].dt.to_period('M') == date.to_period('M')]['Total'].sum() if not div_hasta_fecha.empty else 0
            })
        
        return pd.DataFrame(monthly_performance)
        
    except Exception as e:
        st.warning(f"⚠️ Error al calcular rendimiento mensual: {str(e)}")
        return pd.DataFrame()


def format_currency(amount: float, currency: str = "USD") -> str:
    """Formatear cantidad como moneda."""
    try:
        if currency == "USD":
            return f"${amount:,.2f}"
        elif currency == "MXN":
            return f"${amount:,.2f} MXN"
        else:
            return f"{amount:,.2f} {currency}"
    except:
        return f"{amount}"


def format_percentage(percentage: float, decimals: int = 2) -> str:
    """Formatear porcentaje."""
    try:
        return f"{percentage:.{decimals}f}%"
    except:
        return f"{percentage}%"


def calculate_correlation_matrix(symbols: List[str], start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
    """Calcular matriz de correlación entre instrumentos del portafolio."""
    try:
        import yfinance as yf
        
        if len(symbols) < 2:
            return pd.DataFrame()
        
        if end_date is None:
            end_date = datetime.now()
        
        # Descargar datos de precios
        price_data = pd.DataFrame()
        
        for symbol in symbols:
            try:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    price_data[symbol] = data['Close']
            except:
                continue
        
        if price_data.empty:
            return pd.DataFrame()
        
        # Calcular rendimientos diarios
        returns = price_data.pct_change().dropna()
        
        # Calcular matriz de correlación
        correlation_matrix = returns.corr()
        
        return correlation_matrix
        
    except Exception as e:
        st.warning(f"⚠️ Error al calcular matriz de correlación: {str(e)}")
        return pd.DataFrame()

def calculate_holdings_returns(holdings: pd.DataFrame, current_prices: dict[str, float]):

    try:
        holdings_returns = holdings[['Fecha_Primera_Compra', 'Simbolo', 'Cantidad', 'Precio_Promedio', 'Costo_Total']].copy()
        for row in holdings.index:
            valor_actual = current_prices[holdings.at[row, 'Simbolo']] * holdings.at[row, 'Cantidad']
            pl = valor_actual - holdings.at[row, 'Costo_Total']
            rendimiento = 100 * pl/holdings.at[row, 'Costo_Total']
            holdings_returns.at[row, 'Valor_Actual'] = valor_actual
            holdings_returns.at[row, 'Ganancia/Perdida'] = pl
            holdings_returns.at[row, 'Rendimiento'] = rendimiento

        return holdings_returns

    except Exception as e:
        st.warning(f"⚠️ Error al calcular rendimiento de posiciones: {str(e)}")
        return pd.DataFrame()