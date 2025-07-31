"""
Utilidades para el dashboard de portafolio.
Incluye funciones para carga de datos, cálculos financieros y visualizaciones.
"""

# Importar funciones principales para acceso directo
from .data_loader import (
    conectar_google_sheets,
    load_portfolio_data,
    get_benchmark_data,
    get_available_symbols,
    get_current_prices
)

from .calculations import (

    calculate_current_holdings,
    calculate_portfolio_value,
    calculate_daily_portfolio_values,
    calculate_returns,
    calculate_metrics_summary,
    calculate_sharpe_ratio
)

from .visualizations import (
    plot_portfolio_evolution,
    plot_allocation_pie,
    plot_performance_comparison
)

from .updates import update_portfolio_history

# Versión del paquete
__version__ = "1.0.0"

# Lista de funciones disponibles públicamente
__all__ = [
    # Data loading
    'conectar_google_sheets',
    'load_sheet_data'
    'load_portfolio_data',
    'get_benchmark_data',
    'get_available_symbols',
    'get_current_prices',
    
    # Calculations
    "calculate_current_holdings"
    'calculate_portfolio_value',
    'calculate_daily_portfolio_values',
    'calculate_returns',
    'calculate_metrics_summary',
    'calculate_sharpe_ratio',
    
    # Visualizations
    'plot_portfolio_evolution',
    'plot_allocation_pie',
    'plot_performance_comparison'

    # Updates
    'update_portfolio_history'
]