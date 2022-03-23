from extracting_keywords import *
import streamlit as st

model = init_model()
st.title('Extracting Keyword Model')
text = st.text_area(label = 'Docs: ', height=300)
if text != '':
    predict_keyword = model.predict([text])[0]
    st.text(f"Keywords = {predict_keyword}")