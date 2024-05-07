import pandas as pd
import numpy as np
import seawater as sw
from seawater.library import T90conv
from hampel import hampel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
from tensorflow.keras.models import load_model
import joblib
import argparse
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


def check_data(data):
    # Check if all required columns are present
    required_columns = ['Prof_no', 'Temp_[°C]']
    optional_columns = ['Depth_[m]', 'Pressure_[dbar]']
    
    # Check for invalid syntax or non-float values in all columns
    #try:
    #    data = data.astype(float)
    #except ValueError as e:
    #    return 1 # Invalid syntax or non-float values
    
    if not all(col in data.columns for col in required_columns):
        return 2  # Missing required columns
    
    # Check if at least one optional column is present
    if not any(col in data.columns for col in optional_columns):
        return 3 # At least one optional column is required
 
    return 0  # Data is valid


def process_data(data):
    optional_columns = ['Depth_[m]', 'Pressure_[dbar]']
    # Check if at least one optional column is present
    if not any(col in data.columns for col in optional_columns):
        return 3 # At least one optional column is required
    else:
        # Check if 'Depth' column is missing
        if 'Depth_[m]' not in data.columns:
            # Calculate 'Depth_[m]' column using 'Pressure_[dbar]' and 'Latitude_[deg]'
            data['Depth_[m]'] = sw.eos80.dpth(data['Pressure_[dbar]'], data['Latitude_[deg]'])
        # Check if 'Pressure_' column is missing
        if 'Pressure_[dbar]' not in data.columns:
            # Calculate 'Pressure_' column using 'Depth_[m]' and 'Latitude_[deg]'
            data['Pressure_[dbar]'] = sw.eos80.pres(data['Depth_[m]'], data['Latitude_[deg]'])

    # Check if there are any NaN values
    if data.isnull().values.any():
        # print("Warning; Data contains NaN values")
        data = data.dropna()
        
    # Temperature suspect gradient detection
    # =====================================================================
    #
    #                      Bottom / Top Temp Outliers
    #
    # =====================================================================
    # Temperature suspect gradient detection
    # =====================================================================
    #
    #                      Bottom / Top Temp Outliers
    #
    # =====================================================================
    def Bottom_Top_Temp_Outliers(Data):
        temp_bot_top_outlier=[]
        j=1
        for profile_number in Data.Prof_no.unique():
            profile = Data[Data.Prof_no == profile_number]
            Depth = profile['Depth_[m]'].values
            Temp = profile['Temp_[°C]'].values
            temp_bottom_outlier = []
            temp_top_outlier = []
            nanz = len(np.nonzero(np.isnan(Temp)))
            if (len(np.unique(Temp))>1) & (nanz != len(Temp)):
                # Top ---------------------------------
                h=0
                if np.isnan(Temp[0]):
                    while np.isnan(Temp[h]):
                        h = h + 1
                    starten = h
                else:
                    starten = 0
                T_start = Temp[starten]

                if (T_start < -2) | (T_start > 15) :
                    h=starten
                    while (Temp[h+1] <= (T_start+0.75) ) & ( Temp[h+1] >= (T_start-0.75) ):
                        h = h+1
                        if h==len(Temp)-1:
                            break
                    temp_top_outlier = profile.iloc[[np.arange(starten,h+1)[0]]].index.tolist()

                # Bottom ---------------------------------
                lange = len(Temp)-1;
                h=lange
                if np.isnan(Temp[lange]):
                    while np.isnan(Temp[h]):
                        h = h - 1
                    enden = h.copy()
                else:
                    enden = lange
                T_end = Temp[enden]

                if (T_end < -2) | (T_end > 15) :
                    h=enden
                    while ( Temp[h-1] <= (T_end+0.75) ) & ( Temp[h-1] >= (T_end-0.75) ):
                        h = h-1
                        if h==1:
                            break
                    temp_bottom_outlier = profile.iloc[[np.arange(h,enden+1)[0]]].index.tolist()
                temp_bot_top_outlier.append([temp_top_outlier,temp_bottom_outlier])
                j+=1
        return [item2 for sublist in temp_bot_top_outlier for item in sublist for item2 in item]
    # =====================================================================
    #
    #                   Outliers in mixed layer
    #
    # =====================================================================
    def Traditional_outlier_detection(Data):
        Data['gradientD_T']=np.array(T_Suspect_gradient_D_T(Data))
        data1=Data.loc[(np.array(T_Suspect_gradient_T_D(Data))>=0.5) & (Data['Depth_[m]']<=100)]
        data1.loc[:,'QF_trad'] = 4
        return data1
    def T_Suspect_gradient_D_T(Data):
        unique_profil=Data['Prof_no'].unique()
        d_grad=[]
        data2=[]
        for j in range(len(unique_profil)):
            profil=Data[Data["Prof_no"].isin([unique_profil[j]])].reset_index(drop=True)
            #data2=pd.concat([data2,profil]) 
            data2.append(profil.values) 
            t=profil['Temp_[°C]'].values
            d=profil['Depth_[m]'].values
            i=0
            grad=[-999]
            while i < d.size - 1:
                #case of -999 in temperature
                if -999 in (t[i], t[i+1],d[i + 1], d[i]):
                    grad = np.append(grad, -999) #give this value 1000
                else:
                    if (t[i + 1] - t[i])!=0:
                        grad = np.append(grad, (d[i + 1] - d[i]) / (t[i + 1] - t[i]))
                    else:
                        grad = np.append(grad, -999) #give this value -999
                i+=1
            #grad = np.append(grad,grad[-1])
            d_grad.append(grad)
        d_grad_flat=[item for sublist in d_grad for item in sublist]
        return np.array(d_grad_flat)
    def T_Suspect_gradient_T_D(Data):
        unique_profil=Data['Prof_no'].unique()
        d_grad=[]
        data2=[]
        for j in range(len(unique_profil)):
            profil=Data[Data["Prof_no"].isin([unique_profil[j]])].reset_index(drop=True)
            #data2=pd.concat([data2,profil]) 
            data2.append(profil.values) 
            t=profil['Temp_[°C]'].values
            d=profil['Depth_[m]'].values
            i=0
            grad=[-999]
            while i < d.size - 1:
                #case of -999 in temperature
                if -999 in (t[i], t[i+1],d[i + 1], d[i]):
                    grad = np.append(grad, -999) #give this value 1000
                else:
                    if (d[i + 1] - d[i])!=0:
                        grad = np.append(grad, (t[i + 1] - t[i]) / (d[i + 1] - d[i]))
                    else:
                        grad = np.append(grad, -999) #give this value 1000
                i+=1
            #grad = np.append(grad,grad[-1])
            d_grad.append(grad)
        d_grad_flat=[item for sublist in d_grad for item in sublist]
        return np.array(d_grad_flat)
    # =====================================================================
    #
    #                 Small Temp Outliers below mixed layer
    #
    # =====================================================================
    def Small_Temp_Outliers_below_mixed_layer(Data):
        spike2 = []
        j=1
        for profile_number in Data.Prof_no.unique():
            profile = Data[Data.Prof_no == profile_number]
            Depth = profile['Depth_[m]'].values
            Temp = profile['Temp_[°C]'].values
            temp_gradient = Temp[1:] - Temp[0:-1]
            temp_gradient = np.concatenate([[np.nan],temp_gradient],0)
            if (len(Temp)>7) & (Depth[-1]>100):
                """
                if len(lat)>1: #when add loop
                    break
                """

                find_gradient = temp_gradient.copy()

                # Exclude mixed layer
                flach = np.nonzero(Depth < 100)[0]
                if len(flach)>0:
                    find_gradient[flach] = np.nan

                # --------------------------------------

                windowWidth = round(len(find_gradient)/10) # 10 war o.k.
                if windowWidth < 1:
                    windowWidth = 1

                filtered = hampel(pd.Series(find_gradient), window_size=windowWidth, n=6,imputation=True) # 25 war o.k.

                sd_error = np.nanstd(filtered)
                small_threshold = sd_error # 10 war o.k.   
                small = np.nonzero(((find_gradient > small_threshold) | (find_gradient < -small_threshold)) & (abs(find_gradient) >= 0.2))[0]
                # --------------------------------------

                tips = [];
                h=0;
                tip_count = 0;

                while h < len(small)-1:         

                    grad = temp_gradient.copy()
                    eins = grad[small[h]]
                    zwei = grad[small[h+1]]
                    ten_perc = abs(eins)/2
                    if eins > 0:
                        if zwei < 0:
                            if (abs(zwei) > (abs(eins)-ten_perc)) & (abs(zwei) < (abs(eins)+ten_perc)):
                                tip_this = np.arange(small[h],(small[h+1]))
                                if len(tip_this) <= 3:
                                    tips.append(profile.iloc[tip_this].index.values[0])
                                    tip_count = tip_count + 1
                                else:
                                    if np.std(Temp[tip_this]) <= 0.02:
                                        tips.append(profile.iloc[tip_this].index.values[0])
                                        tip_count = tip_count + 1
                                h = h+2
                            else:
                                h = h+1
                        else:
                            h = h+1


                    if eins < 0:
                        if zwei > 0:
                            if (abs(zwei) > (abs(eins)-ten_perc)) & (abs(zwei) < (abs(eins)+ten_perc)):
                                tip_this = np.arange(small[h],(small[h+1]))
                                if len(tip_this) <= 3:
                                    tips.append(profile.iloc[tip_this].index.values[0])
                                    tip_count = tip_count + 1
                                else:
                                    if np.std(Temp[tip_this]) <= 0.02:
                                        tips.append(profile.iloc[tip_this].index.values[0])
                                        tip_count = tip_count + 1
                                h = h+2
                            else:
                                h = h+1
                        else:
                            h = h+1

                #tips = [item for sublist in tips for item in sublist]
                spike2.append(tips)
            else:
                spike2.append([])
            j+=1
        return [item for sublist in spike2 for item in sublist]
    # =====================================================================
    #
    #                 Temperature gradient detection
    #
    # =====================================================================
    def T_Suspect_gradient(Data):
        unique_profil=Data['Prof_no'].unique()
        d_grad=[]
        data2=[]
        for j in range(len(unique_profil)):
            profil=Data[Data["Prof_no"].isin([unique_profil[j]])].reset_index(drop=True)
            #data2=pd.concat([data2,profil]) 
            data2.append(profil.values) 
            t=profil['Temp_[°C]'].values
            d=profil['Depth_[m]'].values
            i=0
            grad=[-999]
            while i < t.size - 1:
                #case of -999 in temperature
                if -999 in (t[i], t[i+1]):
                    grad = np.append(grad, -999) #give this value -999
                else:
                    if (t[i + 1] - t[i])!=0:
                        grad = np.append(grad, (d[i + 1] - d[i]) / (t[i + 1] - t[i]))
                    else:
                        grad = np.append(grad, -999) #give this value -999
                i+=1
            #grad = np.append(grad,grad[-1])
            d_grad.append(grad)
        d_grad_flat=[item for sublist in d_grad for item in sublist]
        Data['gradient']=np.array(d_grad_flat)
        # get the traditional flages
        Data['QF_trad']=0
        data1=Data.loc[(Data['Depth_[m]']<=100) & ((Data['gradient']>= -0.1) & (Data['gradient']<=1))]
        data2=Data.loc[(Data['Depth_[m]']>100) & ((Data['gradient']>= -0.25) & (Data['gradient']<=5))]
        data3=pd.concat([data1, data2], axis=0)
        data3['QF_trad']=2
        return data3
    # =====================================================================
    #
    #                          Miss temperature
    #
    # =====================================================================
    def Miss_temperature_value(Data):
        data1=Data.loc[(Data['Temp_[°C]']==-999)]
        data1.loc[:,'QF_trad'] = 5
        return data1
    # =====================================================================
    #
    #                      Density_inversion_detection
    #
    # =====================================================================
    '''
    def density_inversion_detection(data):
        unique_profil=data['Prof_no'].unique()
        Density_all = []
        dens_gradient_all=[]
        j=0
        for i in unique_profil:
            profil = data[data.Prof_no == i]
            Salinity = np.array(profil['Salinity_[psu]'])
            Temp = T90conv(profil['Temp_[°C]'])
            Pressure = np.array(profil['Pressure_[dbar]'])
            Density = sw.pden(Salinity, Temp, Pressure)
            dens_gradient = (np.array(Density[1:]) - np.array(Density[0:-1])) # [kg/m^3 per depth unit] # [kg/m^3 per depth unit]
            dens_gradient = np.concatenate([[np.nan],dens_gradient],0).tolist()
            dens_gradient_all.append(dens_gradient)
            Density_all.append(Density)
            j+=1

        dens_gradient_all_flat = [item for sublist in dens_gradient_all for item in sublist]  
        data['dens_gradient'] = dens_gradient_all_flat

        # get the traditional flages
        data1=data.loc[(data['Depth_[m]']<=100) & (data['dens_gradient']<-0.08)]
        data2=data.loc[(data['Depth_[m]']>100) & (data['dens_gradient']<-0.03)]
        data3=pd.concat([data1, data2], axis=0)
        data3['QF_trad']=3
        return data3
    '''
    # Temperature suspect gradient detection
    data.loc[T_Suspect_gradient(data).index]= T_Suspect_gradient(data)
    # Density inversion error
    #data.loc[density_inversion_detection(data).index]= density_inversion_detection(data)
    # Bottom / Top Temp Outliers detection
    data.loc[Bottom_Top_Temp_Outliers(data),'QF_trad']=4
    # Temperature outliers in the upper 100 m 
    data.loc[Traditional_outlier_detection(data).index]= Traditional_outlier_detection(data)
    # Small Temp Outliers below mixed layer
    data.loc[Small_Temp_Outliers_below_mixed_layer(data),'QF_trad']=4
    # Miss values
    data.loc[Miss_temperature_value(data).index]= Miss_temperature_value(data)

    return data  # Processing successful


def predict_data(data, model, scaler):
    # Prediction using machine learning model
    col_names = ['Prof_no','Depth_[m]', 'Temp_[°C]', 'gradientD_T']
    # Placeholder logic for feature scaling (replace with actual logic if needed)
    features = data[col_names].values
    standardized = scaler.transform(features)
    #data[col_names] = standardized
    
    predictions = model.predict(standardized)[:, 0]
    
    # Apply threshold (adjust as needed)
    threshold = 0.84369594

    predictions[predictions <= threshold] = 0
    predictions[predictions > threshold] = 1
    
    return predictions


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process data and save the output to a file.")
    parser.add_argument('--input', required=True, help='input file path')
    parser.add_argument('--output', required=True, help='output file path')
    parser.add_argument('--model', required=True, help='machine learning model path')
    parser.add_argument('--scaler', required=True, help='scaler path')
    args = parser.parse_args()

    # Load machine learning model and scaler
    model = load_model(args.model)
    scaler = joblib.load(args.scaler)

    # Read input data
    data = pd.read_csv(args.input)

    # Check input data
    check_result = check_data(data)
    if check_result == 0:
        # Process data if check is successful
        processed_data = process_data(data)  # Process the data
        predictions = predict_data(processed_data, model, scaler)
        
        # Save the manipulated data to the output file
        processed_data['ML_predictions'] = predictions.astype(int)*processed_data['QF_trad']
        # Use original values of Depth, Temperature, and Gradient for the output
        #processed_data[['Depth_[m]', 'Temp_[°C]', 'gradientD_T']] = data[['Depth_[m]', 'Temp_[°C]', 'gradientD_T']].values
        #rename
        processed_data.rename(columns={'QF_trad': 'Trad_QF', 'ML_predictions': 'ML_QF'}, inplace=True)
        # save
        # processed_data[['Prof_no', 'year', 'month', 'Longitude_[deg]', 'Latitude_[deg]', 'Temp_[°C]', 'Depth_[m]', 'Trad_QF', 'ML_QF']].to_csv(args.output, index=False)
        processed_data[['Trad_QF', 'ML_QF']].to_csv(args.output, index=False)
        #print("Data processing successful. Output saved to", args.output)
    else:
        print(check_result)
        #print("Data check failed. Cannot proceed with further processing.")


    
