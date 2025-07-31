"""
Módulo para visualizaciones del dashboard de portafolio usando Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import streamlit as st

# Configuración de colores
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'portfolio': '#1f77b4',
    'benchmark': '#ff7f0e',
    'positive': '#28a745',
    'negative': '#dc3545',
    'cash': '#6c757d'
}

# Template base para gráficos
PLOTLY_TEMPLATE = 'plotly_white'

def create_base_layout(title: str, height: int = 500) -> dict:
    """Crear layout base para gráficos Plotly."""
    return {
        'title': {
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'family': 'Arial, sans-serif'},
        },
        'height': height,
        'template': PLOTLY_TEMPLATE,
        'showlegend': True,
        'hovermode': 'x unified',
        'margin': dict(l=50, r=50, t=80, b=50)
    }


def plot_portfolio_evolution(portfolio_values: pd.DataFrame, benchmark_data: pd.DataFrame = None, 
                           title: str = "Evolución del Portafolio") -> go.Figure:
    """Graficar evolución temporal del portafolio vs benchmark."""
    
    fig = go.Figure()
    
    if portfolio_values.empty:
        fig.add_annotation(
            text="No hay datos disponibles",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(**create_base_layout(title))
        return fig
    
    try:
        # Preparar datos del portafolio
        portfolio_values = portfolio_values.sort_values('Fecha')
        
        # Calcular valor total estimado
        portfolio_values['Valor_Total'] = portfolio_values['Valor_Posiciones'] + portfolio_values['Efectivo']
        
        # Línea del portafolio
        fig.add_trace(go.Scatter(
            x=portfolio_values['Fecha'],
            y=portfolio_values['Valor_Total'],
            mode='lines',
            name='Portafolio',
            line=dict(color=COLORS['portfolio'], width=3),
            marker=dict(size=6),
            hovertemplate='<b>Portafolio</b><br>' +
                         'Fecha: %{x}<br>' +
                         'Valor: $%{y:,.2f}<extra></extra>'
        ))
        # Agregar anotación en el último punto
        last_x = portfolio_values["Fecha"].iloc[-1]
        last_y = portfolio_values["Valor_Total"].iloc[-1]

        fig.add_annotation(
            x=last_x,
            y=last_y,
            text=f"${last_y:,.2f}",
            showarrow=True,
            arrowhead=1,
            ax=60,
            ay=0,
            font=dict(size=14, color="white"),
            borderwidth=1
        )
        
        # Agregar benchmark si está disponible
        if benchmark_data is not None and not benchmark_data.empty:
            # Normalizar benchmark al valor inicial del portafolio
            if len(portfolio_values) > 0:
                valor_inicial = portfolio_values['Valor_Total'].iloc[0]
                precio_inicial_bench = benchmark_data['Close'].iloc[0]
                benchmark_normalizado = (benchmark_data['Close'] / precio_inicial_bench) * valor_inicial
                
                fig.add_trace(go.Scatter(
                    x=benchmark_data['Date'],
                    y=benchmark_normalizado,
                    mode='lines',
                    name='S&P 500',
                    line=dict(color=COLORS['benchmark'], width=2, dash='dash'),
                    hovertemplate='<b>S&P 500</b><br>' +
                                 'Fecha: %{x}<br>' +
                                 'Valor: $%{y:,.2f}<extra></extra>'
                ))
        
        # Configurar layout
        layout = create_base_layout(title, height=600)
        layout.update({
            'xaxis': {
                'title': 'Fecha',
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
            },
            'yaxis': {
                'title': 'Valor ($)',
                'tickformat': '$,.0f',
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)'
            },
            'legend': {
                'x': 0.02,
                'y': 0.98,
                
            }
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear gráfico de evolución: {str(e)}")
        fig.add_annotation(
            text=f"Error: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        return fig


def plot_allocation_pie(allocation_df: pd.DataFrame, title: str = "Asignación del Portafolio") -> go.Figure:
    """Crear gráfico de torta para asignación de activos."""
    
    fig = go.Figure()
    
    if allocation_df.empty:
        fig.add_annotation(
            text="No hay datos de asignación",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(**create_base_layout(title))
        return fig
    
    try:
        # Preparar datos
        symbols = allocation_df['Simbolo'].tolist()
        values = allocation_df['Valor_Actual'].tolist()
        percentages = allocation_df['Peso_Porcentaje'].tolist()
        
        # Crear colores personalizados
        #colors = px.colors.sequential.Plotly3[:len(symbols)]
        
        # Crear gráfico de pie
        fig.add_trace(go.Pie(
            labels=symbols,
            values=values,
            hole=0.4,  # Crear donut chart
            marker=dict(line=dict(color='white', width=1)),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>' +
                         'Valor: $%{value:,.2f}<br>' +
                         'Porcentaje: %{percent}<br>' +
                         '<extra></extra>'
        ))
        
        # Agregar valor total en el centro
        total_assets = allocation_df['Valor_Actual'].count()
        fig.add_annotation(
            text=f"Activos<br>{total_assets:,.0f}",
            x=0.5, y=0.5,
            font=dict(size=16, color="white"),
            showarrow=False
        )
        
        # Configurar layout
        layout = create_base_layout(title, height=500)
        layout.update({
            'showlegend': True,
            'legend': {
                'orientation': 'v',
                'x': 1.05,
                'y': 1
            }
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear gráfico de asignación: {str(e)}")
        return fig


def plot_performance_comparison(portfolio_returns: pd.DataFrame, benchmark_returns: pd.DataFrame,
                              title: str = "Comparación de Rendimiento") -> go.Figure:
    """Comparar rendimiento del portafolio vs benchmark."""
    
    fig = go.Figure()
    
    try:
        if not portfolio_returns.empty:
            fig.add_trace(go.Scatter(
                x=portfolio_returns['Fecha'],
                y=portfolio_returns['Rendimiento'],
                mode='lines',
                name='Portafolio',
                line=dict(color=COLORS['portfolio'], width=3),
                hovertemplate='<b>Portafolio</b><br>' +
                             'Fecha: %{x}<br>' +
                             'Rendimiento: %{y:.2f}%<extra></extra>'
            ))
            # Agregar anotación en el último punto
            last_x = portfolio_returns["Fecha"].iloc[-1]
            last_y = portfolio_returns["Rendimiento"].iloc[-1]

            fig.add_annotation(
                x=last_x,
                y=last_y,
                text=f"{last_y:,.2f}%",
                showarrow=True,
                arrowhead=1,
                ax=60,
                ay=0,
                font=dict(size=14, color="white"),
                borderwidth=1
            )
        
        if not benchmark_returns.empty:
            fig.add_trace(go.Scatter(
                x=benchmark_returns['Date'],
                y=benchmark_returns['Rendimiento'],
                mode='lines',
                name='SPX',
                line=dict(color=COLORS['benchmark'], width=2),
                hovertemplate='<b>Benchmark</b><br>' +
                             'Fecha: %{x}<br>' +
                             'Rendimiento: %{y:.2f}%<extra></extra>'
            ))
            # Agregar anotación en el último punto
            last_x = benchmark_returns["Date"].iloc[-1]
            last_y = benchmark_returns["Rendimiento"].iloc[-1]

            fig.add_annotation(
                x=last_x,
                y=last_y,
                text=f"{last_y:,.2f}%",
                showarrow=True,
                arrowhead=1,
                ax=60,
                ay=0,
                font=dict(size=14, color="white"),
                borderwidth=1
            )
        
        # Línea de referencia en 0%
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Configurar layout
        layout = create_base_layout(title, height=500)
        layout.update({
            'xaxis': {'title': 'Fecha'},
            'yaxis': {'title': 'Rendimiento (%)', 'ticksuffix': '%'},
            'legend': {'x': 0.02, 'y': 0.98}
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear comparación de rendimiento: {str(e)}")
        return fig


def plot_monthly_returns_heatmap(monthly_returns: pd.DataFrame, 
                                title: str = "Rendimientos Mensuales") -> go.Figure:
    """Crear heatmap de rendimientos mensuales."""
    
    fig = go.Figure()
    
    if monthly_returns.empty:
        fig.add_annotation(
            text="No hay datos de rendimientos mensuales",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        fig.update_layout(**create_base_layout(title))
        return fig
    
    try:
        # Crear heatmap
        fig.add_trace(go.Heatmap(
            z=monthly_returns.values,
            x=monthly_returns.columns,
            y=monthly_returns.index,
            colorscale='RdYlGn',
            zmid=0,
            text=monthly_returns.round(1),
            texttemplate="%{text}%",
            textfont={"size": 10},
            hovertemplate='Año: %{y}<br>' +
                         'Mes: %{x}<br>' +
                         'Rendimiento: %{z:.2f}%<extra></extra>'
        ))
        
        layout = create_base_layout(title, height=400)
        layout.update({
            'xaxis': {'title': 'Mes'},
            'yaxis': {'title': 'Año'}
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear heatmap: {str(e)}")
        return fig


def plot_drawdown_chart(portfolio_values: pd.DataFrame, 
                       title: str = "Drawdown del Portafolio") -> go.Figure:
    """Graficar drawdown del portafolio."""
    
    fig = go.Figure()
    
    if portfolio_values.empty:
        fig.add_annotation(
            text="No hay datos disponibles",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        fig.update_layout(**create_base_layout(title))
        return fig
    
    try:
        # Calcular drawdown
        portfolio_values = portfolio_values.sort_values('Fecha')
        values = portfolio_values['Costo_Invertido'] + portfolio_values['Efectivo']
        
        # Peak running
        peak = values.expanding().max()
        
        # Drawdown
        drawdown = (values - peak) / peak * 100
        
        # Gráfico de área para drawdown
        fig.add_trace(go.Scatter(
            x=portfolio_values['Fecha'],
            y=drawdown,
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color=COLORS['danger'], width=2),
            fillcolor='rgba(220, 53, 69, 0.3)',
            hovertemplate='<b>Drawdown</b><br>' +
                         'Fecha: %{x}<br>' +
                         'Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        # Línea de referencia en 0%
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
        
        layout = create_base_layout(title, height=400)
        layout.update({
            'xaxis': {'title': 'Fecha'},
            'yaxis': {'title': 'Drawdown (%)', 'ticksuffix': '%'},
            'showlegend': False
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear gráfico de drawdown: {str(e)}")
        return fig


def plot_sector_allocation(sector_df: pd.DataFrame, 
                          title: str = "Asignación por Sector") -> go.Figure:
    """Crear gráfico de pastel para asignación por sector."""
    
    fig = go.Figure()
    
    if sector_df.empty:
        fig.add_annotation(
            text="No hay datos de asignación",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(**create_base_layout(title))
        return fig
    
    try:
        # Preparar datos
        symbols = sector_df['Sector'].tolist()
        values = sector_df['Valor'].tolist()
        percentages = sector_df['Porcentaje'].tolist()
        
        # Crear colores personalizados
        #colors = px.colors.qualitative.Set3[:len(symbols)]
        
        # Crear gráfico de pie
        fig.add_trace(go.Pie(
            labels=symbols,
            values=values,
            hole=0.4,  # Crear donut chart
            marker=dict(line=dict(color='white', width=1)),
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>' +
                         'Valor: $%{value:,.2f}<br>' +
                         'Porcentaje: %{percent}<br>' +
                         '<extra></extra>'
        ))
        
        # Agregar valor total en el centro
        total_sectors = sector_df['Valor'].count()
        fig.add_annotation(
            text=f"Sectores<br>{total_sectors:,.0f}",
            x=0.5, y=0.5,
            font=dict(size=16, color="white"),
            showarrow=False
        )
        
        # Configurar layout
        layout = create_base_layout(title, height=500)
        layout.update({
            'showlegend': True,
            'legend': {
                'orientation': 'v',
                'x': 1.05,
                'y': 1
            }
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear gráfico de asignación: {str(e)}")
        return fig


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, 
                           title: str = "Matriz de Correlación") -> go.Figure:
    """Crear heatmap de correlación entre instrumentos."""
    
    fig = go.Figure()
    
    if correlation_matrix.empty:
        fig.add_annotation(
            text="No hay datos de correlación",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        fig.update_layout(**create_base_layout(title))
        return fig
    
    try:
        fig.add_trace(go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.round(3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>' +
                         'Correlación: %{z:.3f}<extra></extra>'
        ))
        
        layout = create_base_layout(title, height=500)
        layout.update({
            'xaxis': {'title': 'Instrumentos'},
            'yaxis': {'title': 'Instrumentos'}
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear matriz de correlación: {str(e)}")
        return fig


def plot_dividend_timeline(dividendos: pd.DataFrame, 
                          title: str = "Cronología de Dividendos") -> go.Figure:
    """Crear gráfico temporal de dividendos recibidos."""
    
    fig = go.Figure()
    
    if dividendos.empty:
        fig.add_annotation(
            text="No hay dividendos registrados",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        fig.update_layout(**create_base_layout(title))
        return fig
    
    try:
        # Agrupar dividendos por mes
        dividendos['Año_Mes'] = dividendos['Fecha'].dt.to_period('M')
        monthly_dividends = dividendos.groupby('Año_Mes')['Total'].sum().reset_index()
        monthly_dividends['Fecha'] = monthly_dividends['Año_Mes'].dt.to_timestamp()
        
        # Gráfico de barras
        fig.add_trace(go.Bar(
            x=monthly_dividends['Fecha'],
            y=monthly_dividends['Total'],
            marker=dict(color=COLORS['success']),
            name='Dividendos',
            hovertemplate='<b>Dividendos</b><br>' +
                         'Mes: %{x}<br>' +
                         'Total: $%{y:,.2f}<extra></extra>'
        ))
        
        layout = create_base_layout(title, height=400)
        layout.update({
            'xaxis': {'title': 'Fecha'},
            'yaxis': {'title': 'Dividendos ($)', 'tickformat': '$,.0f'},
            'showlegend': False
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear cronología de dividendos: {str(e)}")
        return fig


def plot_cash_flow(transacciones: pd.DataFrame, operaciones: pd.DataFrame,
                  title: str = "Flujo de Efectivo") -> go.Figure:
    """Crear gráfico de flujo de efectivo."""
    
    fig = go.Figure()
    
    try:
        all_cash_flows = []
        
        # Procesar transacciones
        if not transacciones.empty:
            for _, trans in transacciones.iterrows():
                monto = trans['Monto'] if trans['Tipo'] == 'Deposito' else -trans['Monto']
                all_cash_flows.append({
                    'Fecha': trans['Fecha'],
                    'Monto': monto,
                    'Tipo': trans['Tipo'],
                    'Categoria': 'Transacción'
                })
        
        # Procesar operaciones (solo el valor neto)
        if not operaciones.empty:
            for _, op in operaciones.iterrows():
                monto = -op['Valor'] if op['Tipo'] == 'Compra' else op['Valor']
                all_cash_flows.append({
                    'Fecha': op['Fecha'],
                    'Monto': monto,
                    'Tipo': op['Tipo'],
                    'Categoria': 'Operación'
                })
        
        if not all_cash_flows:
            fig.add_annotation(
                text="No hay datos de flujo de efectivo",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            fig.update_layout(**create_base_layout(title))
            return fig
        
        # Convertir a DataFrame
        cash_flow_df = pd.DataFrame(all_cash_flows)
        cash_flow_df = cash_flow_df.sort_values('Fecha')
        
        # Separar flujos positivos y negativos
        positive_flows = cash_flow_df[cash_flow_df['Monto'] > 0]
        negative_flows = cash_flow_df[cash_flow_df['Monto'] < 0]
        
        # Agregar flujos positivos
        if not positive_flows.empty:
            fig.add_trace(go.Bar(
                x=positive_flows['Fecha'],
                y=positive_flows['Monto'],
                name='Entradas',
                marker=dict(color=COLORS['success']),
                hovertemplate='<b>Entrada</b><br>' +
                             'Fecha: %{x}<br>' +
                             'Monto: $%{y:,.2f}<extra></extra>'
            ))
        
        # Agregar flujos negativos
        if not negative_flows.empty:
            fig.add_trace(go.Bar(
                x=negative_flows['Fecha'],
                y=negative_flows['Monto'],
                name='Salidas',
                marker=dict(color=COLORS['danger']),
                hovertemplate='<b>Salida</b><br>' +
                             'Fecha: %{x}<br>' +
                             'Monto: $%{y:,.2f}<extra></extra>'
            ))
        
        # Línea de referencia en 0
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
        
        layout = create_base_layout(title, height=500)
        layout.update({
            'xaxis': {'title': 'Fecha'},
            'yaxis': {'title': 'Monto ($)', 'tickformat': '$,.0f'},
            'barmode': 'relative'
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear gráfico de flujo de efectivo: {str(e)}")
        return fig


def create_metrics_gauge(value: float, title: str, min_val: float = 0, max_val: float = 100,
                        threshold_good: float = 70, threshold_bad: float = 30) -> go.Figure:
    """Crear gráfico de gauge para métricas."""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [min_val, threshold_bad], 'color': "lightgray"},
                {'range': [threshold_bad, threshold_good], 'color': "yellow"},
                {'range': [threshold_good, max_val], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    
    return fig

def plot_holdings_performance(holdings_returns: pd.DataFrame, 
                             title: str = "Rendimiento por Instrumento") -> go.Figure:
    """Crear gráfico de barras horizontales para rendimiento de holdings."""
    
    fig = go.Figure()
    
    if holdings_returns.empty or 'Rendimiento' not in holdings_returns.columns:
        fig.add_annotation(
            text="No hay datos de rendimiento de holdings",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(**create_base_layout(title))
        return fig
    
    try:
        # Preparar datos - ordenar por rendimiento
        holdings_sorted = holdings_returns.sort_values('Rendimiento', ascending=True)
        
        # Crear colores basados en rendimiento (verde para positivo, rojo para negativo)
        colors = []
        for rendimiento in holdings_sorted['Rendimiento']:
            if rendimiento > 0:
                colors.append(COLORS['positive'])
            elif rendimiento < 0:
                colors.append(COLORS['negative'])
            else:
                colors.append('gray')
        
        # Crear gráfico de barras horizontales
        fig.add_trace(go.Bar(
            x=holdings_sorted['Rendimiento'],
            y=holdings_sorted['Simbolo'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=0)
            ),
            text=[f'{r:+.2f}%' for r in holdings_sorted['Rendimiento']],
            textposition='outside',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{y}</b><br>' +
                         'Rendimiento: %{x:,.2f}%<br>' +
                         'Valor Actual: $%{customdata[0]:,.2f}<br>' +
                         'Ganancia/Pérdida: $%{customdata[1]:,.2f}<br>' +
                         '<extra></extra>',
            customdata=holdings_sorted[['Valor_Actual', 'Ganancia/Perdida']].values,
            name='Rendimiento'
        ))
        
        # Línea de referencia en 0%
        fig.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.7, line_width=2)
        
        # Configurar layout
        layout = create_base_layout(title, height=max(400, len(holdings_sorted) * 40))
        layout.update({
            'xaxis': {
                'title': 'Rendimiento (%)',
                'ticksuffix': '%',
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'zeroline': True,
                'zerolinecolor': 'rgba(128,128,128,0.5)',
                'zerolinewidth': 2,
                'range': [holdings_sorted['Rendimiento'].min() * 1.1, holdings_sorted['Rendimiento'].max() * 1.2]
            },
            'yaxis': {
                'title': 'Símbolo',
                'showgrid': False,
                'categoryorder': 'array',
                'categoryarray': holdings_sorted['Simbolo'].tolist()
            },
            'showlegend': False,
            'margin': dict(l=80, r=150, t=80, b=50),
            'hovermode': 'y'
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear gráfico de rendimiento de holdings: {str(e)}")
        fig.add_annotation(
            text=f"Error: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        return fig

def plot_holdings_pl(holdings_returns: pd.DataFrame, 
                             title: str = "Ganancias/Pérdidas por Instrumento") -> go.Figure:
    """Crear gráfico de barras horizontales para rendimiento de holdings."""
    
    fig = go.Figure()
    
    if holdings_returns.empty or 'Ganancia/Perdida' not in holdings_returns.columns:
        fig.add_annotation(
            text="No hay datos de ganancias o pérdidas de holdings",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(**create_base_layout(title))
        return fig
    
    try:
        # Preparar datos - ordenar por rendimiento
        holdings_sorted = holdings_returns.sort_values('Ganancia/Perdida', ascending=True)
        
        # Crear colores basados en rendimiento (verde para positivo, rojo para negativo)
        colors = []
        for pl in holdings_sorted['Ganancia/Perdida']:
            if pl > 0:
                colors.append(COLORS['positive'])
            elif pl < 0:
                colors.append(COLORS['negative'])
            else:
                colors.append('gray')
        
        # Crear gráfico de barras horizontales
        fig.add_trace(go.Bar(
            x=holdings_sorted['Ganancia/Perdida'],
            y=holdings_sorted['Simbolo'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=0)
            ),
            text=[f'{r:+.2f}' for r in holdings_sorted['Ganancia/Perdida']],
            textposition='outside',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{y}</b><br>' +
                         'Rendimiento: %{customdata[0]:,.2f}%<br>' +
                         'Valor Actual: $%{customdata[1]:,.2f}<br>' +
                         'Ganancia/Pérdida: $%{customdata[2]:,.2f}<br>' +
                         '<extra></extra>',
            customdata=holdings_sorted[['Rendimiento', 'Valor_Actual', 'Ganancia/Perdida']].values,
            name='Rendimiento'
        ))
        
        # Línea de referencia en 0%
        fig.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.7, line_width=2)
        
        # Configurar layout
        layout = create_base_layout(title, height=max(400, len(holdings_sorted) * 40))
        layout.update({
            'xaxis': {
                'title': 'Ganancias/Perdidas',
                'ticksuffix': '',
                'showgrid': True,
                'gridcolor': 'rgba(128,128,128,0.2)',
                'zeroline': True,
                'zerolinecolor': 'rgba(128,128,128,0.5)',
                'zerolinewidth': 2,
                'range': [holdings_sorted['Ganancia/Perdida'].min() * 1.1, holdings_sorted['Ganancia/Perdida'].max() * 1.2]
            },
            'yaxis': {
                'title': 'Símbolo',
                'showgrid': False,
                'categoryorder': 'array',
                'categoryarray': holdings_sorted['Simbolo'].tolist()
            },
            'showlegend': False,
            'margin': dict(l=80, r=150, t=80, b=50),
            'hovermode': 'y'
        })
        
        fig.update_layout(layout)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error al crear gráfico de rendimiento de holdings: {str(e)}")
        fig.add_annotation(
            text=f"Error: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False
        )
        return fig