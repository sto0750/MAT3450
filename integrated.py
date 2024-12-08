import pandas as pd

# Load models
from ETS import ETS_model

# Load data
districts = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구', '노원구',
             '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '성북구', '송파구',
             '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구']

district_of_interest = districts[1]

df = pd.read_csv('seoul_data_csv/{}_data_filled.csv'.format(district_of_interest))
print(district_of_interest, "예상 월별 미세먼지 \t PM10 \t PM2.5")
ETS = ETS_model(df)
ETS.preset()
ETS.actual_predict(term=5)