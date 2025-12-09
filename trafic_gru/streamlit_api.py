import streamlit as st
import requests

st.title("Traffic GRU Model Inference")
st.markdown("""
This application allows you to input traffic data and get predictions from a GRU model.
""")

seq_input=[]
with st.form("input_form"):
    st.subheader("Input Traffic Data Sequence")
    for i in range(24):
        st.markdown(f"### Time Step {i+1}")
        col1,col2,col3,col4,col5,col6,col7= st.columns(7)
        with col1:
            temp=st.number_input("Temp(°K)",value=290.0,min_value=250.0,max_value=320.0,step=0.1,key=f"temp_{i}")
        with col2:
            rain=st.number_input("Rain",value=0.0,min_value=0.0,max_value=100.0,step=0.1,key=f"rain_{i}")
        with col3:
            snow=st.number_input("Snow",value=0.0,min_value=0.0,max_value=100.0,step=0.1,key=f"snow_{i}")
        with col4:
            clouds=st.slider("Clouds",value=0.0,min_value=0.0,max_value=100.0,step=0.1,key=f"clouds_{i}")
        with col5:
            hour=i
        with col6:
            dayofweek=st.selectbox("Day of Week",options=list(range(0,7)),index=1,key=f"dayofweek_{i}")
        with col7:
            month=st.selectbox("Month",options=list(range(1,13)),index=0,key=f"month_{i}")
        seq_input.append([temp, rain, snow, clouds, hour, dayofweek, month])
    
    submitted = st.form_submit_button("Get Prediction")
api_url="http://localhost:8000/predict" # Replace with your actual API endpoint

if submitted:
    try:
        response = requests.post(api_url, json={"seq": seq_input})
        if response.status_code == 200:
            prediction = response.json()
            st.success("Prediction received successfully!")
            st.json(prediction)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)
        
