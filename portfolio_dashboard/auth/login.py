import streamlit as st
import bcrypt
import time

# Configuración de usuarios
USERS = {
    st.secrets["users"]["admin"]["name"]: {
        "password": st.secrets["auth"]["admin_password"],
        "role": st.secrets["users"]["admin"]["role"]
    },
    st.secrets["users"]["usuario"]["name"]: {
        "password": st.secrets["auth"]["user_password"],
        "role": st.secrets["users"]["usuario"]["role"]
    }
}

def verify_credentials(username, password):
    """Verificar credenciales"""
    if username in USERS:
        if username in USERS:
            stored = USERS[username]["password"]
            return bcrypt.checkpw(password.encode(), stored.encode())
    return False

def login_form():
    """Mostrar formulario de login"""
    
    st.markdown("# 🔐 Acceso al Dashboard")
    st.markdown("---")
    
    with st.form("login_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            username = st.text_input("👤 Usuario", placeholder="Ingresa tu usuario")
            password = st.text_input("🔒 Contraseña", type="password", placeholder="Ingresa tu contraseña")
            
            submit_button = st.form_submit_button("🚀 Iniciar Sesión", use_container_width=True)
            
            if submit_button:
                if username and password:
                    if verify_credentials(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_data = USERS[username]
                        st.session_state.login_time = time.time()
                        st.success("✅ ¡Login exitoso!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Usuario o contraseña incorrectos")
                else:
                    st.warning("⚠️ Por favor completa todos los campos")
    
    st.markdown("</div>", unsafe_allow_html=True)

def logout():
    """Cerrar sesión"""
    for key in ['authenticated', 'username', 'user_data', 'login_time']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def check_session_timeout(timeout_minutes=60):
    """Verificar timeout de sesión"""
    if 'login_time' in st.session_state:
        elapsed = time.time() - st.session_state.login_time
        if elapsed > (timeout_minutes * 60):
            st.warning("⏰ Sesión expirada. Por favor inicia sesión nuevamente.")
            logout()
            return False
    return True

def show_user_info():
    """Mostrar información del usuario logueado"""
    if st.session_state.get('authenticated'):
        user_data = st.session_state.get('user_data', {})
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"👋 **Bienvenido {user_data.get('name', 'Usuario')}**")
        
        with col2:
            if user_data.get('role') == 'admin':
                st.badge("👑 Admin", type="secondary")
        
        with col3:
            if st.button("🚪 Cerrar Sesión"):
                logout()

def require_auth():
    """Verificar autenticación"""
    # Verificar si está autenticado
    if not st.session_state.get('authenticated', False):
        return False
    
    # Verificar timeout
    if not check_session_timeout():
        return False
    
    return True
