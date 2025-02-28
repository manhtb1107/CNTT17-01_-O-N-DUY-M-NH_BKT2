import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Bước 1. Thu thập dữ liệu (giả sử đã có file Excel, đọc từ file)
file_path = 'CNTT17-01_ĐOÀN DUY MẠNH_BKT2.xlsx'
df = pd.read_excel(file_path)

# In dữ liệu ban đầu
print("Dữ liệu ban đầu:")
print(df)

# Bước 2. Làm sạch dữ liệu
# 2.1. Xử lý định dạng ngày tháng
df['Ngày tháng'] = pd.to_datetime(df['Ngày tháng'], errors='coerce')

# 2.2. Xử lý giá trị âm trong 'Lượng mưa (mm)'
df['Lượng mưa (mm)'] = df['Lượng mưa (mm)'].apply(lambda x: x if x >= 0 else None)
mean_rainfall = df['Lượng mưa (mm)'].mean()
df['Lượng mưa (mm)'] = df['Lượng mưa (mm)'].fillna(mean_rainfall)

# 2.3. Xử lý giá trị thiếu
df['Nguồn mưa'] = df['Nguồn mưa'].fillna('Không rõ')
df['Mức độ nghiêm trọng'] = df['Mức độ nghiêm trọng'].fillna(df['Mức độ nghiêm trọng'].mode()[0])

# In dữ liệu sau khi làm sạch
print("\nDữ liệu sau khi làm sạch:")
print(df)

# Bước 3. EDA - Phân tích dữ liệu khám phá
print("\nThống kê cơ bản của các cột số:")
print(df.describe())

# Biểu đồ cột lượng mưa theo địa điểm
plt.figure(figsize=(10, 6))
plt.bar(df['Địa điểm'], df['Lượng mưa (mm)'])
plt.xticks(rotation=45)
plt.title('Lượng mưa theo địa điểm')
plt.xlabel('Địa điểm')
plt.ylabel('Lượng mưa (mm)')
plt.tight_layout()
plt.show()

# Biểu đồ phân phối lượng mưa
plt.figure(figsize=(8, 6))
plt.hist(df['Lượng mưa (mm)'], bins=10, edgecolor='black')
plt.title('Phân phối lượng mưa')
plt.xlabel('Lượng mưa (mm)')
plt.ylabel('Số lần xảy ra')
plt.show()

# Bước 4. Xây dựng mô hình
# Chuẩn bị dữ liệu
# Chuyển đổi cột categorical thành số
le = LabelEncoder()
df['Địa điểm'] = le.fit_transform(df['Địa điểm'])
df['Nguồn mưa'] = le.fit_transform(df['Nguồn mưa'])
df['Mức độ nghiêm trọng'] = le.fit_transform(df['Mức độ nghiêm trọng'])

# Tách biến độc lập (X) và phụ thuộc (y)
X = df["Lượng mưa (mm)"].values.reshape(-3, 1)
y = df["Nguồn mưa"].values

#Xây dựng mô hình
model = LinearRegression()
model.fit(X, y)

# Dự đoán giá trị
y_pred = model.predict(X)

# Tính toán các hệ số hồi quy
beta_0 = model.intercept_  # Hệ số chặn
beta_1 = model.coef_[0]    # Hệ số dốc

# Đánh giá mô hình
r2 = r2_score(y, y_pred)  # Hệ số xác định R^2
mse = mean_squared_error(y, y_pred)  # Sai số bình phương trung bình (MSE)

# Phân tích kết quả
print("\nPhân tích kết quả:")
print("Mô hình Linear Regression có hiệu suất:", "Tốt" if r2 > 0.7 else "Cần cải thiện")
print("Đề xuất cải thiện:")
print("- Thử nghiệm các mô hình khác như Random Forest hoặc Gradient Boosting.")
print("- Kiểm tra và thêm các đặc trưng mới nếu cần.")
print("- Tăng kích thước dữ liệu hoặc xử lý outliers nếu có.")
# Bước 5 : # Vẽ biểu đồ
plt.scatter(X, y, color="blue", label="Dữ liệu thực tế")
plt.plot(X, y_pred, color="red", label="Dự đoán (hồi quy)")
plt.title("Hồi quy tuyến tính: Số lượng và Tổng doanh thu")
plt.xlabel("Số Lượng")
plt.ylabel("Tổng Doanh Thu")
plt.legend()
plt.grid(True)
plt.show()