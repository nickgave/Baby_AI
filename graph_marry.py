from matplotlib import pyplot as plt
import openpyxl as exel
import numpy as np
import pandas as pd
import os

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

folder_path = './data/'

file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
data_list = []

# 각 파일을 읽고 numpy.array로 변환한 후, 리스트에 저장
df = pd.read_csv('./data/marry_num.csv')  # CSV 파일 읽기
data_array = df.values[:]  # pandas DataFrame을 numpy array로 변환

#데이터 정리
x_nums = np.array(data_array[:,0])
y_nums = np.array(data_array[:,1])
dataframe = pd.DataFrame({
    '연도': x_nums,
    '사교육비': y_nums,
}); print(dataframe)


#x축, y축 설정
x = dataframe['연도'].values
y = dataframe['사교육비'].values

#y=mx+n에서 최소제곱법을 이용해 m값, n값 구하기
m = np.cov(x,y,ddof=1)[0,1]/np.cov(x,y,ddof=1)[0,0]
n = np.mean(y)-m*np.mean(x)

# 2050년까지 예측
future_dates = range(2022,2031)
future_predictions = m * future_dates + n

# 모든 날짜와 예측값 결합
all_dates = np.append(dataframe['연도'].values, future_dates)
all_predictions = np.append(10**(m * x + n), future_predictions)

print(f"Regression Equation: {m:.5f}x+{n:.5f}")

#일반적인 그래프 그리기
plt.figure(figsize=(10,8), dpi=75, num="sagyoyuk chubang")
plt.title("결혼 건수의 변화")
plt.xlabel("연도 (년)")
plt.ylabel("결혼 건수 (건)")
plt.scatter(future_dates[-1], future_predictions[-1], c='red', edgecolors='red', s=10)
plt.plot(dataframe['연도'], dataframe['사교육비'], color='blue', label='실제 데이터')
plt.plot(dataframe['연도'], m * x + n, color='red', linestyle='-', label='회귀 직선')
plt.plot(future_dates, future_predictions, color='red', linestyle='--', label='회귀 직선 [예측값]')
plt.annotate(f'{future_dates[-1]}\nPredicted: {future_predictions[-1]:.2f}건',
             xy=(future_dates[-1], future_predictions[-1]),
             xytext=(future_dates[-1], future_predictions[-1] * 1.08),
             arrowprops=dict(facecolor='black', shrink=0.05),
             horizontalalignment='center')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

plt.show()

m = int(input())
n = -m * x[-1] + y[-1]
print(f"Changed Equation: {m:.5f}x+{n:.5f}")
while True:
    a = int(input())
    if a == 'q': break
    print(m * a + n, '건')