import streamlit as st
from auth.login import require_auth, login_form

from utils.calculations import calculate_current_holdings, calculate_daily_portfolio_values, calculate_holdings_returns, calculate_metrics_summary, calculate_portfolio_allocation, calculate_portfolio_value, calculate_returns, calculate_sector_allocation
from utils.data_loader import *
from utils.visualizations import *
from utils.updates import update_portfolio_history
import streamlit as st


def show_dashboard():
    st.title("Portafolio de inversiones")

    update_portfolio_history()

    operaciones, transacciones, dividendos, daily_portfolio_values = load_portfolio_data()
    benchmark = get_benchmark_data('^SPX', start_date='2025-03-23')

    symbols = get_available_symbols(operaciones)
    current_prices = get_current_prices(symbols)

    holdings = calculate_current_holdings(operaciones)
    holdings_returns = calculate_holdings_returns(holdings, current_prices)

    st.subheader('Cambio del portafolio')

    plot_evolution = plot_portfolio_evolution(daily_portfolio_values)
    plot_performance = plot_performance_comparison(daily_portfolio_values, benchmark)

    tab1, tab2 = st.tabs(['Valor', 'Rendimiento'])
    
    with tab1:
        st.plotly_chart(plot_evolution, width=True)
    with tab2:
        st.plotly_chart(plot_performance, width=True)

    st.subheader('Distribución')

    portfolio_allocation = calculate_portfolio_allocation(holdings, current_prices)
    sector_allocation = calculate_sector_allocation(holdings, current_prices)

    plot_portfolio_allocation = plot_allocation_pie(portfolio_allocation)
    plot_sectors = plot_sector_allocation(sector_allocation)

    tab1, tab2 = st.tabs(['Activos', 'Sectores'])

    with tab1:
        st.plotly_chart(plot_portfolio_allocation, width=True)
    with tab2:
        st.plotly_chart(plot_sectors, width=True)

    st.subheader("Posiciones")

    plot_holdings = plot_holdings_performance(holdings_returns)
    plot_hold_pl = plot_holdings_pl(holdings_returns)

    tab1, tab2, tab3 = st.tabs(['Ganancia/Pérdida', 'Rendimiento', 'Data'])
    with tab1:
        st.plotly_chart(plot_hold_pl)
    with tab2:
        st.plotly_chart(plot_holdings)
    with tab3:
        st.write(holdings_returns)

    st.subheader('Movimientos')

    tab1, tab2, tab3 = st.tabs(['Operaciones', 'Transacciones', 'Dividendos'])
    with tab1:
        st.write(operaciones)
    with tab2: 
        st.write(transacciones)
    with tab3:
        st.write(dividendos)

def main():
    st.set_page_config(
        page_title="Dashboard de Portafolio",
        initial_sidebar_state='collapsed',
        layout='wide'
    )
    hide_sidebar = """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            [data-testid="collapsedControl"] {
                display: none;
            }
        </style>
    """
    #st.markdown(hide_sidebar, unsafe_allow_html=True)

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    # Verificar autenticación
    if not require_auth():
        login_form()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()