import pandas as pd
import streamlit as st
import pickle
import base64
import numpy as np
from io import BytesIO
import pandas as pd
import streamlit as st
import pickle
import base64
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# Print library versions


def download_link(object_to_download, download_filename):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{download_filename}">Download Results</a>'

# Function to predict solubility
def predict_mmp(data0):
    df =data0
    IRIS0 = df
    IRIS = np.array(IRIS0)
    H2S = np.array(IRIS0['x_H2S (%)'], dtype=np.float64) ** 0.8
    xH2S = np.array(IRIS0['x_H2S (%)'], dtype=np.float64)
    N2 = np.array(IRIS0['x_N2 (%)'], dtype=np.float64)
    xN2 = np.array(IRIS0['x_N2 (%)'], dtype=np.float64)
    Co2 = np.array(IRIS0['x_Co2 (%)'], dtype=np.float64) ** 1.38
    xCo2 = np.array(IRIS0['x_Co2 (%)'], dtype=np.float64)
    C1 = np.array(IRIS0['x_C1 (%)'], dtype=np.float64) ** 0.7
    xC1 = np.array(IRIS0['x_C1 (%)'], dtype=np.float64)
    C2 = np.array(IRIS0['x_C2 (%)'], dtype=np.float64) ** 0.8
    xC2 = np.array(IRIS0['x_C2 (%)'], dtype=np.float64)
    C3 = np.array(IRIS0['x_C3 (%)'], dtype=np.float64) ** 0.8
    xC3 = np.array(IRIS0['x_C3 (%)'], dtype=np.float64)
    C4 = np.array(IRIS0['x_C4 (%)'], dtype=np.float64) ** 0.8
    xC4 = np.array(IRIS0['x_C4 (%)'], dtype=np.float64)
    C5 = np.array(IRIS0['x_C5 (%)'], dtype=np.float64)
    xC5 = np.array(IRIS0['x_C5 (%)'], dtype=np.float64)
    C6 = np.array(IRIS0['x_C6 (%)'], dtype=np.float64)
    xC6 = np.array(IRIS0['x_C6 (%)'], dtype=np.float64)
    C7p = np.array(IRIS0['x_C7+ (%)'], dtype=np.float64) ** 0.38
    xC7p = np.array(IRIS0['x_C7+ (%)'], dtype=np.float64)
    MWC7p = np.array(IRIS0['MW c7+'], dtype=np.float64) ** 0.9
    MWC7p0 = np.array(IRIS0['MW c7+'], dtype=np.float64)
    Tres = np.array(IRIS0['Tres (F)'], dtype=np.float64)
    MW_oil1 = (xH2S * 34.1 + xCo2 * 44.01 + xC1 * 16.04 + xC2 * 30.07 + xN2 * 28.0134 \
               + xC3 * 44.1 + xC4 * 58.16 + xC5 * 72.15 \
               + xC6 * 86.18 + xC7p * MWC7p0) / 100
    MW_oil = (MW_oil1) ** -2
    MW_ap_C7p1 = MWC7p0 * xC7p / 100
    MW_ap_C7p = MW_ap_C7p1 ** -1.9
    frn1 = (1 + xC1 + xN2) / (1 + xCo2 + xH2S + xC2 + xC3 + xC4 + xC5 + xC6)
    EVP1 = np.exp(8.243 / (1 + 0.002177 * (Tres)) - 10.91)
    frn = (frn1) ** 0.63
    Tsq = ((Tres)) ** 1.9
    EVP = (EVP1) / 14.7
    SG_calc0 = SG_calc0 = 1.106352054 / (46.23006224 / MWC7p0 + 1.090283159)
    SG_calc1 = 0.134462445 + 0.214592184 * SG_calc0 + 0.703011117 * SG_calc0 * SG_calc0 + 0.010846788 * np.exp(SG_calc0)
    A0 = -9092.137474
    A1 = 31.28911371
    A2 = 5.32589172
    A3 = -3.744553796
    A4 = -3.744840184
    A5 = 34.5695143
    A6 = 24.45732707
    A7 = 14.2445869
    A8 = 0.538538391
    A9 = -3.915630434
    A10 = 600.9850745
    A11 = 446.2606576
    A12 = 17.75623621
    A13 = -1328.354546
    A14 = 6407937.97
    A15 = 39.14271794
    A16 = -0.002677839
    A17 = 78781.43976
    Part1 = A0 + A1 * H2S + A2 * Co2 + A3 * N2 + A4 * C1 + A5 * C2 + A6 * C3
    Part2 = A7 * C4 + C5 * A8 + C6 * A9 + C7p * A10 + A11 * frn
    Part3 = A12 * (MWC7p) * (1 + A13 * (MWC7p0 ** -2.8) * ((xC7p / 100) ** -1.9))
    Part4 = A14 * (MW_oil) + A15 * Tres + A15 * A16 * Tsq + A17 * np.exp((8.243 / (1 + 0.002177 * (Tres)) - 10.91))
    Pred = Part1 + Part2 + Part3 + Part4
    MMP = IRIS0['MMP Exp (Psia)']
    Inputs = pd.DataFrame([H2S, N2, Co2, C1, C2, C3, C4, C5, C6, C7p, MWC7p, Tres,MW_oil, MW_ap_C7p, frn, Tsq, EVP, SG_calc1, Pred, MMP]).transpose()
    Inputs_array = np.array(Inputs.iloc[:, 0:19])
    MMP_Actual_psia = np.array(Inputs.iloc[:, 19])
    with open("finalized_MMP_CO2_model_2nf_feb_2023.pkl", 'rb') as f:
        model = pickle.load(f)
    MMP_Predicted = model.predict(Inputs_array)
    # correction on RF predicted MMP
    B0 = 1.13499602e-07
    B1 = -7.09652915e-04
    B2 = 2.64376527e+00
    B3 = -1.10136778e+03
    B4 = 6.67976946e-05
    B5 = 2.74878063e-02
    MMP_Predicted_final_psia = (B0 * (MMP_Predicted ** 3) + B1 * (MMP_Predicted ** 2) + B2 * MMP_Predicted + B3) / (1 + B4 * MMP_Predicted) + B5
    results = IRIS0.copy()
    results['MMP_Predicted (psia)'] = MMP_Predicted_final_psia
    return results
html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">UH Pure CO2 MMP Calculator  </h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)

st.header(
    "Department of Petroleum Engineering: Interaction of Phase Behavior and Flow in Porous Media ([IPBFPM](https://dindoruk.egr.uh.edu/)) Consortium.")

st.markdown('<style>h2{color: red;}</style>', unsafe_allow_html=True)
st.subheader(
    "Product Description -  Calculates the Minimum Miscibility Pressure (psia) for Pure CO2 gas injection..")
st.markdown('<style>h2{color: red;}</style>', unsafe_allow_html=True)
st.subheader(
    "[Download Input Template File.](https://drive.google.com/file/d/1S60FPJcZbqxhCGJ3xTUMhniTqZK0jHC2/view)")
# st.markdown("[Input Template File Link](https://drive.google.com/file/d/1HNyZjobmTEBcWfk0C2cmClQfahTONrX1/view?usp=sharing)",unsafe_allow_html=True)


file = st.file_uploader("Upload the CSV file", type=['csv'])

if file is not None:
    # Load the data
    data = pd.read_csv(file)
    # Display the loaded data
    st.subheader('Loaded Data:')
    st.write(data)
    # Call the predict function
    results = predict_mmp(data)

    # Display the result table
    if st.button('Predict'):
        st.write(results)

    # Download the results as csv
    if st.button('Download Results'):
        csv_data = results.to_csv(index=False)
        tmp_download_link = download_link(csv_data, 'CO2_MMP_Results.csv')
        st.markdown(tmp_download_link, unsafe_allow_html=True)

st.subheader("Developed by Utkarsh Sinha, Dr. Birol Dindoruk and Dr. M.Y. Soliman")

st.subheader(
    "[Based on Work in Ref: Sinha, U., Dindoruk, B., & Soliman, M. (2021). Prediction of CO2 Minimum Miscibility Pressure Using an Augmented Machine-Learning-Based Model. SPE Journal, 1-13. (SPE-200326-PA)](https://onepetro.org/SJ/article-abstract/26/04/1666/460403/Prediction-of-CO2-Minimum-Miscibility-Pressure?redirectedFrom=fulltext)")

from PIL import Image

image = Image.open('uhlogo.jpg')
st.image(image, caption='A product of University of Houston')

# print('result===',result)