import streamlit as st
import pickle
import numpy as np
import os
from streamlit_option_menu import option_menu
#set up page configuration for streamlit

st.set_page_config(page_title='Industrial copper',initial_sidebar_state='expanded',layout='wide')


class option():
    # Value get from data
    country_value=[25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0,
            79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
    application_value=[2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 
                        27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 
                        59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
    status_value=['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
    'Wonderful', 'Revised', 'Offered', 'Offerable']
    # Mapping Value
    status_dict={'Won':1, 'Draft':2, 'To be approved':3, 'Lost':0, 'Not lost for AM':4,
                                'Wonderful':5, 'Revised':6, 'Offered':7, 'Offerable':8}
    item_type_value=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    # Encoded value
    item_dict={'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}

    product_value=[611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 
            640665, 164141591, 164336407, 164337175, 929423819, 1282007633, 
            1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662, 
            1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 
            1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]
    

st.title(":blue[Industrial Copper Modeling]")
# Data Interpretation
select=option_menu('',options=["Selling Price","Status"],
                                icons=["cash", "toggles"],
                                orientation='horizontal',)
if select=='Selling Price':
    st.write("To predict the selling price of copper, please provide the following information:")
    st.write('')
# Selling Price prediction data
    with st.form('prediction'):
            col1,col2=st.columns(2)
            with col1:
                item_date=st.date_input(label='Item Date',format='DD/MM/YYYY')

                customer=st.text_input(label='customer ID,(min:12458.0 & max:2147483650.0)')

                country=st.selectbox(label='Country', options=option.country_value)

                status=st.selectbox(label='status',options=option.status_value)

                Item_type=st.selectbox(label='Item_type',options=option.item_type_value)

                application=st.selectbox(label='Application',options=option.application_value)

            with col2:
                delivery_date=st.date_input(label='Delivery Date',format='DD/MM/YYYY')

                width=st.number_input(label='Width',min_value=1.0,max_value=29990000.0,value=1.0)

                product_ref=st.selectbox(label='Product_ref',options=option.product_value)

                quantity_log = st.number_input(label='Quantity Tons',min_value=0.00001,max_value=1000000000.0,value=1.0)

                thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)

                button=st.form_submit_button('PREDICT',use_container_width=True)

    if button:
            # Not fillung all the columns Command
            if not all([item_date, delivery_date, country, Item_type, application, product_ref,
                        customer, status, quantity_log, width, thickness_log]):
                st.error("Please fill in all required fields.")

            else:
                # using pickle file to Predict data
                with open('regression_model.pkl','rb') as files:
                    predict_model=pickle.load(files)
                status=option.status_dict[status]
                item_type=option.item_dict[Item_type]

                delivery_time_taken=abs((item_date - delivery_date).days) # To get the difference between the item and delivery date

                quantity_log=np.log(quantity_log) # Data is given as input in logrithmic form
                thickness_log=np.log(thickness_log) # Data is given as input in logrithmic form

                user_data=np.array([[customer, country, status, item_type ,application, width, product_ref,
                                    delivery_time_taken, quantity_log, thickness_log ]]) # Giving input to predict data using picle file
                
                pred=predict_model.predict(user_data)

                selling_price=np.exp(pred[0]) # selling price is in log for form we use exp to retransform the Data

                st.subheader(f":green[Predicted Selling Price :] {selling_price:.2f}")

if select == 'Status':
        st.write("To predict the status of copper, please provide the following information:")
        st.write('')
        # Predict the status using classification algorithm

        with st.form('classifier'):
            col1,col2=st.columns(2)
            with col1:
                # Data to be collected for process the status

                item_date=st.date_input(label='Item Date',format='DD/MM/YYYY')

                customer=st.text_input(label='customer ID,(min:12458.0 & max:2147483650.0)')

                country=st.selectbox(label='Country', options=option.country_value)

                product_ref=st.selectbox(label='Product_ref',options=option.product_value)

                Item_type=st.selectbox(label='Item_type',options=option.item_type_value)

                application=st.selectbox(label='Application',options=option.application_value)

            with col2:
                delivery_date=st.date_input(label='Delivery Date',format='DD/MM/YYYY')

                width=st.number_input(label='Width',min_value=1.0,max_value=29990000.0,value=1.0)

                quantity_log = st.number_input(label='Quantity Tons',min_value=0.00001,max_value=1000000000.0,value=1.0)

                thickness_log = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)

                selling_price=st.number_input(label='Selling Price',min_value=0.1)

                
                
                button=st.form_submit_button('PREDICT',use_container_width=True)

        if button:

            if not all([item_date, delivery_date, country, Item_type, application, product_ref,
                        customer,quantity_log, width, thickness_log,selling_price]):
                st.error("Please fill in all required fields.")

            else:
                # Pickle file already store the model in it

                with open('classification_model.pkl','rb') as files:
                    model=pickle.load(files)
    
                item_type=option.item_dict[Item_type]

                delivery_time_taken=abs((item_date - delivery_date).days) # To get the difference between the item and delivery date

                quantity_log=np.log(quantity_log) #Data is given as input in logrithmic form
                thickness_log=np.log(thickness_log) #Data is given as input in logrithmic form
                selling_price_log=np.log(selling_price) #Data is given as input in logrithmic form

                user_data=np.array([[customer, country, item_type ,application, width, product_ref,
                                    delivery_time_taken, quantity_log, thickness_log, selling_price_log ]])
                
                status=model.predict(user_data)

                if status==1:
                    st.subheader(f":green[Status of the copper : ] Won")

                else:
                    st.subheader(f":red[Status of the copper :] Lost")

