import datetime
import numpy as np
import sklearn

#### Prep Functions
def prep_locations(prepared_data):
  locations = prepared_data["current_location"].values

  longitudes = np.zeros(len(locations))
  latitudes = np.zeros(len(locations))
  for i in range(len(locations)):
    longitudes[i], latitudes[i] = locations[i].split(",")[0][2:-2], locations[i].split(",")[1][2:-2]
  prepared_data["longitude"] = longitudes
  prepared_data["latitude"] = latitudes
  prepared_data = prepared_data.drop("current_location", axis=1)

def date_to_timestamp(date_str): #2021-12-18
  date = datetime.datetime(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]))
  return date.timestamp()

def prep_dates(prepared_data):
  dates = prepared_data["pcr_date"].values

  dates_num = np.zeros(len(dates))

  for i in range(len(dates)):
    dates_num[i] = date_to_timestamp(dates[i])
  prepared_data["pcr_date_timestamp"] = dates_num
  
  prepared_data = prepared_data.drop("pcr_date", axis=1)

## Normalizers

def StandardNormalize(df, df_new, columns):
  scaler = sklearn.preprocessing.StandardScaler()
  scaler.fit(df[columns])
  df_new[columns] = scaler.transform(df_new[columns])

def MinMaxNormalize(df, df_new, columns):
  scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
  scaler.fit(df[columns])
  df_new[columns] = scaler.transform(df_new[columns])


### Prepare Data
def prepare_data(training_data, new_data):
  prepared_data = new_data.copy()
  training_data_copy = training_data.copy()

  #Remove Patient Id
  prepared_data = prepared_data.drop("patient_id", axis=1)

  #Replace NaN values for household_income:
  median = training_data["household_income"].median()
  prepared_data["household_income"] = prepared_data["household_income"].fillna(median)

  #Prepare Blood Types
  prepared_data["SpecialProperty"] = prepared_data["blood_type"].isin(["O+", "B+"])
  prepared_data = prepared_data.drop("blood_type", axis=1)

  #Prepare Longitude and Latitude
  prep_locations(prepared_data)
  prep_locations(training_data_copy)

  #Prepare Date:
  prep_dates(prepared_data)
  prep_dates(training_data_copy)

  #Normalization
  StandardNormalize(training_data_copy, prepared_data, ["PCR_01", "PCR_02", "PCR_05", "PCR_06", "PCR_07", "PCR_08", "age", "weight", "num_of_siblings", "happiness_score", "conversations_per_day", "household_income","sugar_levels", "longitude"])
  MinMaxNormalize(training_data_copy, prepared_data, ["PCR_10", "PCR_03", "PCR_04", "PCR_09", "sport_activity", "latitude", "pcr_date_timestamp"])
  return prepared_data