from app.core.ui_utils import DataHandler
import streamlit as st

session_data_handler = DataHandler()
st.session_state['data_handler'] = session_data_handler
session_data_handler.run()

