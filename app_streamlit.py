import pickle
from datetime import datetime

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


@st.cache_resource
def load_models():
    def load_obj(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

    label_encoder = load_obj(r'encoders_scalers/label_encoder.pickle')
    standard_scaler = load_obj(r'encoders_scalers/standard_scaller.pickle')
    model = load_model(r'models_cache/deep_5.keras')

    return label_encoder, standard_scaler, model


def main():
    st.set_page_config(page_title='Weather Classification using Deep Learning', page_icon='🌦️', layout='wide')

    with open('README.md', 'r', encoding='UTF-8') as f:
        st.title(f.read())

    st.divider()

    Input_Temp_C = 9.3
    Input_Dew_Point_Temp_C = 3.3
    Input_Rel_Hum_Percent = 68.0
    Input_Wind_Speed_km_h = 13.0
    Input_Visibility_km = 25.0
    Input_Press_kPa = 101.0

    st.image('dataset-cover.png')

    with st.container():
        cols = [st.columns(2) for _ in range(4)]

        Input_date = cols[0][0].date_input("Select date")
        Input_time = cols[0][1].time_input("Select time")

        Input_Temp_C = cols[1][0].number_input(label='Temp (C)', value=Input_Temp_C)
        Input_Dew_Point_Temp_C = cols[1][1].number_input(label='Dew Point Temp (C)', value=Input_Dew_Point_Temp_C)

        Input_Rel_Hum_Percent = cols[2][0].number_input(label='Rel Hum (%)', value=Input_Rel_Hum_Percent, min_value=0.0,
                                                        max_value=100.0)
        Input_Wind_Speed_km_h = cols[2][1].number_input(label='Wind Speed (km/h)', value=Input_Wind_Speed_km_h)

        Input_Visibility_km = cols[3][0].number_input(label='Visibility (km)', value=Input_Visibility_km)
        Input_Press_kPa = cols[3][1].number_input(label='Press (kPa)', value=Input_Press_kPa)


        if st.button('Predict', type='primary', use_container_width=True):
            y_labeled = predict( datetime.combine(Input_date, Input_time), Input_Temp_C, Input_Dew_Point_Temp_C, Input_Rel_Hum_Percent, Input_Wind_Speed_km_h, Input_Visibility_km, Input_Press_kPa)

            for class_name, percent in y_labeled:
                st.subheader(f"{class_name}: {int(percent)}%")


def predict(date_time, temp, dew_point, rel_humidity, wind_speed, visibility, pressure):
    label_encoder, standard_scaler, model = load_models()

    df = pd.DataFrame(
        {
            'Date/Time': [date_time],
            'Temp_C': [temp],
            'Dew Point Temp_C': [dew_point],
            'Rel Hum_%': [rel_humidity],
            'Wind Speed_km/h': [wind_speed],
            'Visibility_km': [visibility],
            'Press_kPa': [pressure]
        }
    )

    df['month'] = df['Date/Time'].apply(lambda row: row.month)
    df['day'] = df['Date/Time'].apply(lambda row: (row.month - 1) * 30.0 + row.day)

    def month_to_season(month):
        month = month.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            raise f'Invalid month: {month}'

    df['year_quarter'] = df['Date/Time'].apply(month_to_season)

    df['am-!pm'] = df['Date/Time'].apply(lambda x: float(x.hour in range(6, 18)))

    df.drop(['Date/Time'], axis=1, inplace=True)

    df['year_quarter_Autumn'] = (df['year_quarter'] == 'Autumn').astype(int)
    df['year_quarter_Spring'] = (df['year_quarter'] == 'Spring').astype(int)
    df['year_quarter_Summer'] = (df['year_quarter'] == 'Summer').astype(int)
    df['year_quarter_Winter'] = (df['year_quarter'] == 'Winter').astype(int)
    df.drop(['year_quarter'], axis=1, inplace=True)


    FEATURES_INPUT_NUMERICAL = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km',
                                'Press_kPa', 'day']
    df.loc[:,FEATURES_INPUT_NUMERICAL] = standard_scaler.transform(df[FEATURES_INPUT_NUMERICAL])

    # st.write(df.values)

    y = model.predict(df.values)
    # st.write(y)

    y_labeled = []
    for i, class_name in enumerate(label_encoder.classes_):
        y_labeled.append( (class_name, 100.0 * float(y[0][i])) )

    # st.write(y_labeled)
    return y_labeled


if __name__ == "__main__":
    main()
