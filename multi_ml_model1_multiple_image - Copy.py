import streamlit as st
import pickle
import numpy as np
import pdb
import pandas as pd
import datetime
import io
import json
import torch
import cv2
import matplotlib.pyplot as plt
import pytesseract        # added code
from pytesseract import image_to_string # added code 
import glob
import pandas as pd
import re
import streamlit as st
import os
import imageio
from PIL import Image,ImageEnhance



today = datetime.date.today()
year = today.year


st.set_page_config(layout="wide",page_icon="🧊",initial_sidebar_state="auto")
st.sidebar.title("Price Prediction Tool")
st.sidebar.image('data:image/png;base64,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')


# removing streamlit 
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)
user_menu = st.sidebar.selectbox('Select Options',('CV (CS)','CV (NCS)','2W (CS)','2W (NCS)','FE (CS)','FE (NCS)','4W (NCS)','4W (CS)','ANPR'))


if user_menu == "CV (CS)":
	df = pd.read_csv('model_data.csv',encoding='latin-1',low_memory=False)
	loaded_model=pickle.load(open('regressor_CV_CS.pkl','rb'))

	
	st.title("Commercial Vehicle Price Prediction Tool (CS)")
	st.image('https://i.ytimg.com/vi/y0XA5VJRXXY/maxresdefault.jpg')


	col1,col2,col3,col4 = st.columns(4)
	col5,col6,col7,col8 = st.columns(4)


	make  =df['Make_Clean'].unique().tolist()
	make.insert(0,"Select Make")
	Make_Clean = col1.selectbox('Select Make',make,index = 0)


	model = df['Model_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	model.insert(0,"Select Model")
	Model_Clean = col2.selectbox('Select Model',model,index = 0)


	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).drop_duplicates().tolist()
	# mk_yr.insert(0,'Select Year')
	# MAKE_YEAR = col3.selectbox('Select Make Year',mk_yr,index = 0)
	MAKE_YEAR = col3.number_input('Enter Make Year in YYYY format', min_value=None, max_value=None, step=1, value=year)	


	variant = df['Variant_Clean'].loc[df['Model_Clean'] == Model_Clean].drop_duplicates().tolist()
	variant.insert(0,'Select Variant')
	Variant_Clean = col4.selectbox('Select Variant',variant,index = 0)


	fuel = df['Fuel_Clean'].loc[df['Variant_Clean'] == Variant_Clean].drop_duplicates().tolist()
	fuel.insert(0,'Select Fuel Type')
	Fuel_Clean = col5.selectbox('Select Fuel',fuel,index = 0)


	Meter_Reading = col7.number_input('Enter Meter Reading',min_value=None)


	state = df['CV_State_Clean'].unique().tolist()
	state.insert(0,'Select State')
	CV_State_Clean = col8.selectbox('Select State',state,index = 0)


	cust_seg = df['SELLER_SEGMENT'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	cust_seg.insert(0,'Select Segmentation')
	SELLER_SEGMENT = col6.selectbox("Select Customer Segmentation",cust_seg,index = 0)

	X = pd.DataFrame(columns=['MAKE_YEAR','Make_Clean','Model_Clean','Variant_Clean','Fuel_Clean','CV_State_Clean','SELLER_SEGMENT','Meter_Reading'],data=np.array([MAKE_YEAR,Make_Clean,Model_Clean,Variant_Clean,Fuel_Clean,CV_State_Clean,SELLER_SEGMENT,Meter_Reading]).reshape(1,8))

	if st.button('Get the Best Price'):
		predicted_amount = loaded_model.predict(X)
		st.subheader(f"The Estimated CV Price is ₹ {predicted_amount[0]:.2f}")


elif user_menu == "CV (NCS)":

	df = pd.read_csv('Final_CV_NCS.csv',encoding='latin-1',low_memory=False)
	loaded_model=pickle.load(open('regressor.pkl','rb'))

	st.title("Commercial Vehicle Price Prediction Tool (NCS)")
	st.image('https://i.ytimg.com/vi/y0XA5VJRXXY/maxresdefault.jpg')

	col1,col2,col3,col4 = st.columns(4)
	col5,col6,col7,col8 = st.columns(4)


	make  =df['Make_Clean'].unique().tolist()
	make.insert(0,"Select Make")
	Make_Clean = col1.selectbox('Select Make',make,index = 0)


	model = df['Model_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	model.insert(0,"Select Model")
	Model_Clean = col2.selectbox('Select Model',model,index = 0)

	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).drop_duplicates().tolist()
	# mk_yr.insert(0,'Select Year')
	# MAKE_YEAR = col3.selectbox('Select Make Year',mk_yr,index = 0)
	MAKE_YEAR = col3.number_input('Enter Make Year in YYYY format', min_value=None, max_value=None, step=1, value=year)	


	variant = df['Variant_Clean'].loc[df['Model_Clean'] == Model_Clean].drop_duplicates().tolist()
	variant.insert(0,'Select Variant')
	Variant_Clean = col4.selectbox('Select Variant',variant,index = 0)


	fuel = df['Fuel_Clean'].loc[df['Variant_Clean'] == Variant_Clean].drop_duplicates().tolist()
	fuel.insert(0,'Select Fuel Type')
	Fuel_Clean = col5.selectbox('Select Fuel',fuel,index = 0)

	Registration = col6.text_input("Enter Registration")

	Meter_Reading = col7.number_input('Enter Meter Reading',min_value=None)

	state = df['CV_State_Clean'].unique().tolist()
	CV_State_Clean = col8.selectbox('Select State',state)


	X = pd.DataFrame(columns=['MAKE_YEAR','Make_Clean','Model_Clean','Variant_Clean','Fuel_Clean','CV_State_Clean','Meter_Reading'],data=np.array([MAKE_YEAR,Make_Clean,Model_Clean,Variant_Clean,Fuel_Clean,CV_State_Clean,Meter_Reading]).reshape(1,7))


	if st.button('Get the Best Price'):
		predicted_amount = loaded_model.predict(X)
		st.subheader(f"The Estimated CV Price is ₹ {predicted_amount[0]:.2f}")


elif user_menu == "2W (CS)":
	df = pd.read_csv('model_data_2w_cs.csv',encoding='latin-1',low_memory=False)
	loaded_model=pickle.load(open('regressor_2w_cs.pkl','rb'))

	st.title("Two Wheeler Price Prediction (CS)")
	st.image('https://3.bp.blogspot.com/-pMbQWXysi6M/Wmb0sKjtz-I/AAAAAAAAMhk/kLlMlocxmEco9tzuv_sM2kvk1eWGzm13wCLcBGAs/s1600/used%2Bscooter.png')


	col1,col2,col3 = st.columns(3)
	col4,col5,col6,col7 = st.columns(4)

	make = df['Make_Clean'].unique().tolist()
	make.insert(0,"Select Make")
	Make_Clean = col1.selectbox('Select Make',make,index=0)

	model = df['Model_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	model.insert(0,"Select Model")
	Model_Clean = col2.selectbox('Select Model',model,index=0)

	variant = df['Variant_Clean'].loc[df['Model_Clean'] == Model_Clean].drop_duplicates().tolist()
	variant.insert(0,'Select Variant')
	Variant_Clean = col3.selectbox('Select Variant',variant,index = 0)

	fuel = df['Fuel_Clean'].loc[df['Variant_Clean'] == Variant_Clean].drop_duplicates().tolist()
	fuel.insert(0,'Select Fuel Type')
	Fuel_Clean = col4.selectbox('Select Fuel',fuel,index = 0)

	state = df['State_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	state.insert(0,'Select State')
	State_Clean = col5.selectbox('Select State',state,index = 0)

	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).unique().tolist()
	# mk_yr.insert(0,'Select Year')
	# MAKE_YEAR = col6.selectbox('Select Make Year',mk_yr,index = 0)
	MAKE_YEAR = col6.number_input('Enter Make Year in YYYY format', min_value=None, max_value=None, step=1, value=year)	


	cust_seg = df['Customer_Segmentation'].unique().tolist()
	cust_seg.insert(0,'Select Segmentation')
	Customer_Segmentation = col7.selectbox("Select Segmentation",cust_seg,index = 0)


	X = pd.DataFrame(columns=['Make_Clean','Model_Clean','Variant_Clean','Fuel_Clean','State_Clean','MAKE_YEAR','Customer_Segmentation'],data=np.array([Make_Clean,Model_Clean,Variant_Clean,Fuel_Clean,State_Clean,MAKE_YEAR,Customer_Segmentation]).reshape(1,7))


	if st.button('Get the Best Price'):
		predicted_amount = loaded_model.predict(X)
		st.subheader(f"The Estimated 2W Price is ₹ {predicted_amount[0]:.2f}")


elif user_menu == "2W (NCS)":
	df = pd.read_csv('model_data_2w_ncs.csv',encoding='latin-1',low_memory=False)
	loaded_model=pickle.load(open('regressor_2w_ncs.pkl','rb'))

	st.title("Two Wheeler Price Prediction (NCS)")
	st.image('https://3.bp.blogspot.com/-pMbQWXysi6M/Wmb0sKjtz-I/AAAAAAAAMhk/kLlMlocxmEco9tzuv_sM2kvk1eWGzm13wCLcBGAs/s1600/used%2Bscooter.png')


	col1,col2,col3 = st.columns(3)
	col4,col5,col6 = st.columns(3)

	make = df['Make_Clean'].unique().tolist()
	make.insert(0,"Select Make")
	Make_Clean = col1.selectbox('Select Make',make,index=0)

	model = df['Model_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	model.insert(0,"Select Model")
	Model_Clean = col2.selectbox('Select Model',model,index=0)

	variant = df['Variant_Clean'].loc[df['Model_Clean'] == Model_Clean].drop_duplicates().tolist()
	variant.insert(0,'Select Variant')
	Variant_Clean = col3.selectbox('Select Variant',variant,index = 0)

	fuel = df['Fuel_Clean'].loc[df['Variant_Clean'] == Variant_Clean].drop_duplicates().tolist()
	fuel.insert(0,'Select Fuel Type')
	Fuel_Clean = col4.selectbox('Select Fuel',fuel,index = 0)

	state = df['State_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	state.insert(0,'Select State')
	State_Clean = col5.selectbox('Select State',state,index = 0)


	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).unique().tolist()
	# mk_yr.insert(0,'Select Year')
	# MAKE_YEAR = col6.selectbox('Select Make Year',mk_yr,index = 0)
	MAKE_YEAR = col6.number_input('Enter Make Year in YYYY format', min_value=None, max_value=None, step=1, value=year)	

	X = pd.DataFrame(columns=['Make_Clean','Model_Clean','Variant_Clean','Fuel_Clean','State_Clean','MAKE_YEAR'],data=np.array([Make_Clean,Model_Clean,Variant_Clean,Fuel_Clean,State_Clean,MAKE_YEAR]).reshape(1,6))


	if st.button('Get the Best Price'):
		predicted_amount = loaded_model.predict(X)
		st.subheader(f"The Estimated 2W Price is ₹ {predicted_amount[0]:.2f}")
		

elif user_menu == "FE (CS)":
	df = pd.read_csv('model_data_fe_cs.csv',encoding='latin-1')
	loaded_model=pickle.load(open('regressor_fe_cs.pkl','rb'))

	st.title("Farmer's Equipment Price Prediction (CS)")
	st.image('https://2.bp.blogspot.com/-LlpNLl9GpKw/WovqbQsY15I/AAAAAAAAMl0/clpEbYsCnXQ5UvuKGZTf-avDHS0gFBxmQCLcBGAs/s1600/used%2Btractor.png')


	col1,col2,col3,col4 = st.columns(4)
	col5,col6,col7,col8 = st.columns(4)

	make = df['Make_Clean'].unique().tolist()
	make.insert(0,"Select Make")
	Make_Clean = col1.selectbox('Select Make',make,index=0)

	model = df['Model_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	model.insert(0,"Select Model")
	Model_Clean = col2.selectbox('Select Model',model,index=0)

	variant = df['Variant_Clean'].loc[df['Model_Clean'] == Model_Clean].drop_duplicates().tolist()
	variant.insert(0,'Select Variant')
	Variant_Clean = col3.selectbox('Select Variant',variant,index = 0)


	fuel = df['Fuel_Clean'].loc[df['Variant_Clean'] == Variant_Clean].drop_duplicates().tolist()
	fuel.insert(0,'Select Fuel')
	Fuel_Clean = col4.selectbox('Select Fuel',fuel,index = 0)
	
	
	state = df['STATE_MAPPED'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	state.insert(0,'Select State')
	STATE_MAPPED = col5.selectbox('Select State',state,index = 0)


	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).unique().tolist()
	# mk_yr.insert(0,'Select Year')
	# MAKE_YEAR = col6.selectbox('Select Make Year',mk_yr,index = 0)
	MAKE_YEAR = col6.number_input('Enter Make Year in YYYY format', min_value=None, max_value=None, step=1, value=year)	


	cust_seg = df['SELLER_SEGMENT'].loc[df['Variant_Clean'] == Variant_Clean].drop_duplicates().tolist()
	cust_seg.insert(0,'Select Segmentation')
	SELLER_SEGMENT = col7.selectbox('Select Segmentation',cust_seg,index = 0)

	METERREADING = col8.number_input('Enter Meter Reading',min_value=None)

	X = (pd.DataFrame(columns=['MAKE_YEAR','Make_Clean','Model_Clean','Variant_Clean','Fuel_Clean','STATE_MAPPED','SELLER_SEGMENT','METERREADING'],data=np.array([MAKE_YEAR,Make_Clean,Model_Clean,Variant_Clean,Fuel_Clean,STATE_MAPPED,SELLER_SEGMENT,METERREADING]).reshape(1,8)))


	if st.button('Get the Best Price'):
		predicted_amount = loaded_model.predict(X)
		st.subheader(f"The Estimated FE Price is ₹ {predicted_amount[0]:.2f}")	


elif user_menu == "FE (NCS)":
	df = pd.read_csv('FE_NCS_.csv',encoding='latin-1')
	loaded_model=pickle.load(open('regressor_FE_NCS.pkl','rb'))

	st.title("Farmer's Equipment Price Prediction (NCS)")
	st.image('https://2.bp.blogspot.com/-LlpNLl9GpKw/WovqbQsY15I/AAAAAAAAMl0/clpEbYsCnXQ5UvuKGZTf-avDHS0gFBxmQCLcBGAs/s1600/used%2Btractor.png')


	col1,col2,col3 = st.columns(3)
	col4,col5,col6,col7 = st.columns(4)

	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).unique().tolist()
	# mk_yr.insert(0,'Select Year')
	# MAKE_YEAR = col1.selectbox('Select Make Year',mk_yr,index = 0)
	MAKE_YEAR = col1.number_input('Enter Make Year in YYYY format', min_value=None, max_value=None, step=1, value=year)	

	make = df['Make_Clean'].unique().tolist()
	make.insert(0,"Select Make")
	Make_Clean = col2.selectbox('Select Make',make,index=0)

	model = df['Model_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	model.insert(0,"Select Model")
	Model_Clean = col3.selectbox('Select Model',model,index=0)

	variant = df['Variant_Clean'].loc[df['Model_Clean'] == Model_Clean].drop_duplicates().tolist()
	variant.insert(0,'Select Variant')
	Variant_Clean = col4.selectbox('Select Variant',variant,index = 0)


	state = df['CV_State_Clean'].unique().tolist()
	state.insert(0,'Select State')
	CV_State_Clean = col5.selectbox('Select State',state,index = 0)
	

	fuel = df['Fuel_Clean'].loc[df['Variant_Clean'] == Variant_Clean].drop_duplicates().tolist()
	fuel.insert(0,'Select Fuel')
	Fuel_Clean = col6.selectbox('Select Fuel',fuel,index = 0)


	METERREADING = col7.number_input('Enter Meter Reading',min_value=None)

	
	X = pd.DataFrame(columns=['MAKE_YEAR','Make_Clean','Model_Clean','Variant_Clean','Fuel_Clean','CV_State_Clean','METERREADING'],data=np.array([MAKE_YEAR,Make_Clean,Model_Clean,Variant_Clean,Fuel_Clean,CV_State_Clean,METERREADING]).reshape(1,7))


	if st.button('Get the Best Price'):
		predicted_amount = loaded_model.predict(X)
		st.subheader(f"The Estimated FE Price is ₹ {predicted_amount[0]:.2f}")


elif user_menu == "ANPR":
	pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
	model = torch.hub.load("D:/pricex_model_api/deep_learning/yolov5-master", 'custom', path = "D:/pricex_model_api/deep_learning/best.pt", source='local', force_reload=True)
	upload_path = "uploads/"

	@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
	def yolomodel(img,count):

		frame = cv2.imread(img)
		detections = model(frame)
		results = detections.pandas().xyxy[0].to_dict(orient="records")        
		for result in results:
			con = result['confidence']
			cs = result['class']
			if(con>0.45):
				x1 = int(result['xmin'])
				y1 = int(result['ymin'])
				x2 = int(result['xmax'])
				y2 = int(result['ymax'])
				
				cropped_img = frame[y1:y2,x1:x2]
				cv2.imwrite(f'D:/pricex_model_api/multi_model_app/download/{count}.jpg',cropped_img)
				img=Image.open(f'D:/pricex_model_api/multi_model_app/download/{count}.jpg')
				img_contr_obj=ImageEnhance.Contrast(img)
				e_img=img_contr_obj.enhance(factor=0.85)

			try:
				text = pytesseract.image_to_string(e_img, config='-l eng --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890') # added code
				raw = re.sub("[^A-Z0-9 -]", "", text)
				final_result = re.sub(r'[\W_]+', '', raw)
				return final_result

			except:
				return f"Not Readable"


	st.title("""Automatic Number Plate Recognition""")
	st.image('https://zatpark.com/uploads/blog/blog-what-is-anpr.jpg')
	st.info('Supports all popular image formats 📷 - PNG, JPG, JPEG')
	uploaded_file = st.file_uploader("Upload Image of car's number plate 🚓", type=["png","jpg","bmp","jpeg","jfif"],accept_multiple_files=True)

	nbr_plate = []	

	for i in uploaded_file:
		count = 0
		if i is not None:
			with open(os.path.join(upload_path,i.name),"wb") as f:
				f.write((i).getbuffer())
			
			with st.spinner(f"Working... 💫"):
				uploaded_image = os.path.abspath(os.path.join(upload_path,i.name))
				res = yolomodel(uploaded_image,count)
				nbr_plate.append(res)
				count += 1

	
	nbr_plate = set(nbr_plate)
	nbr_plate = list(nbr_plate)

	df = pd.Series(nbr_plate,name='Number Plate')

	hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

	# Inject CSS with Markdown
	st.markdown(hide_table_row_index, unsafe_allow_html=True)

	st.table(df)


elif user_menu == "4W (NCS)":
	df = pd.read_csv('4w_ncs.csv',encoding='latin-1')
	loaded_model=pickle.load(open('FInal_regressor_4w_NCS.pkl','rb'))

	st.title("Four Wheeler Price Prediction (NCS)")
	st.image('https://www.samil.in/images/banner-img1.jpg')


	col1,col2,col3 = st.columns(3)
	col4,col5,col6,col7 = st.columns(4)

	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).unique().tolist()
	# mk_yr.insert(0,'Select Year')
	# MAKE_YEAR = col1.selectbox('Select Make Year',mk_yr,index = 0)
	MAKE_YEAR = col1.number_input('Enter Make Year in YYYY format', min_value=None, max_value=None, step=1, value=year)	

	make = df['Make_Clean'].unique().tolist()
	make.insert(0,"Select Make")
	Make_Clean = col2.selectbox('Select Make',make,index=0)


	model = df['Model_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	model.insert(0,"Select Model")
	Model_Clean = col3.selectbox('Select Model',model,index=0)


	variant = df['Variant_Clean'].loc[df['Model_Clean'] == Model_Clean].drop_duplicates().tolist()
	variant.insert(0,'Select Variant')
	Variant_Clean = col4.selectbox('Select Variant',variant,index = 0)


	state = df['STATE_MAPPED'].unique().tolist()
	state.insert(0,'Select State')
	STATE_MAPPED = col5.selectbox('Select State',state,index = 0)
	

	fuel = df['Fuel_Clean'].loc[df['Variant_Clean'] == Variant_Clean].drop_duplicates().tolist()
	fuel.insert(0,'Select Fuel')
	Fuel_Clean = col6.selectbox('Select Fuel',fuel,index = 0)


	METERREADING = col7.number_input('Enter Meter Reading',min_value=None)

	
	X = pd.DataFrame(columns=['MAKE_YEAR','Make_Clean','Model_Clean','Variant_Clean','Fuel_Clean','STATE_MAPPED','METERREADING'],data=np.array([MAKE_YEAR,Make_Clean,Model_Clean,Variant_Clean,Fuel_Clean,STATE_MAPPED,METERREADING]).reshape(1,7))


	if st.button('Get the Best Price'):
		predicted_amount = loaded_model.predict(X)
		st.subheader(f"The Estimated 4 Wheeler Price is ₹ {predicted_amount[0]:.2f}")


elif user_menu == "4W (CS)":
	df = pd.read_csv('4w_cs.csv',encoding='latin-1')
	loaded_model=pickle.load(open('FInal_4W_CS.pkl','rb'))

	st.title("Four Wheeler Price Prediction (CS)")
	st.image('https://www.samil.in/images/banner-img1.jpg')


	col1,col2,col3,col4 = st.columns(4)
	col5,col6,col7,col8 = st.columns(4)

	# mk_yr = df['MAKE_YEAR'].sort_values(ascending=False).unique().tolist()
	# mk_yr.insert(0,'Select Year')
	# MAKE_YEAR = col1.selectbox('Select Make Year',mk_yr,index = 0)
	MAKE_YEAR = col1.number_input('Enter Make Year in YYYY format', min_value=None, max_value=None, step=1, value=year)	

	make = df['Make_Clean'].unique().tolist()
	make.insert(0,"Select Make")
	Make_Clean = col2.selectbox('Select Make',make,index=0)


	model = df['Model_Clean'].loc[df['Make_Clean'] == Make_Clean].drop_duplicates().tolist()
	model.insert(0,"Select Model")
	Model_Clean = col3.selectbox('Select Model',model,index=0)


	variant = df['Variant_Clean'].loc[df['Model_Clean'] == Model_Clean].drop_duplicates().tolist()
	variant.insert(0,'Select Variant')
	Variant_Clean = col4.selectbox('Select Variant',variant,index = 0)


	state = df['CV_State_Clean'].unique().tolist()
	state.insert(0,'Select State')
	CV_State_Clean = col5.selectbox('Select State',state,index = 0)


	SELLER_SEGMENT = df['SELLER_SEGMENT'].unique().tolist()
	state.insert(0,'Select Segment')
	SELLER_SEGMENT = col6.selectbox('Select Segment',SELLER_SEGMENT,index = 0)
	

	fuel = df['Fuel_Clean'].loc[df['Variant_Clean'] == Variant_Clean].drop_duplicates().tolist()
	fuel.insert(0,'Select Fuel')
	Fuel_Clean = col7.selectbox('Select Fuel',fuel,index = 0)


	METERREADING = col8.number_input('Enter Meter Reading',min_value=None)

	
	X = pd.DataFrame(columns=['MAKE_YEAR','Make_Clean','Model_Clean','Variant_Clean','Fuel_Clean','CV_State_Clean','SELLER_SEGMENT','METERREADING'],data=np.array([MAKE_YEAR,Make_Clean,Model_Clean,Variant_Clean,Fuel_Clean,CV_State_Clean,SELLER_SEGMENT,METERREADING]).reshape(1,8))


	if st.button('Get the Best Price'):
		predicted_amount = loaded_model.predict(X)
		st.subheader(f"The Estimated 4 Wheeler Price is ₹ {predicted_amount[0]:.2f}")