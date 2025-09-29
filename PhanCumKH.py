import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Đọc và khám phá dữ liệu
# Đảm bảo file customer.csv ở cùng thư mục với main.py
try:
    df = pd.read_csv("customers.csv")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'customer.csv'. Vui lòng kiểm tra lại đường dẫn.")
    exit()

print("5 dòng đầu tiên:")
print(df.head())
print("\nThông tin dữ liệu:")
print(df.info())

# 2. Tiền xử lý dữ liệu
# Xử lý giá trị thiếu trong cột 'Profession'
df['Profession'].fillna(df['Profession'].mode()[0], inplace=True)

# Lựa chọn các đặc trưng số để phân cụm
X = df[['Age', 'Annual Income ($)', 'Spending Score (1-100)', 'Work Experience', 'Family Size']]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nDữ liệu sau khi chuẩn hóa:")
print(pd.DataFrame(X_scaled, columns=X.columns).head())

# 3. Tìm số cụm tối ưu (k) bằng Elbow Method
sse = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sse.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, sse, 'bx-')
plt.xlabel('Số cụm (k)')
plt.ylabel('SSE')
plt.title('Elbow Method để tìm số cụm tối ưu')
plt.show()

# Dựa vào biểu đồ Elbow, chọn k=5 (hoặc giá trị bạn thấy phù hợp)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nĐã phân cụm khách hàng thành {n_clusters} nhóm.")

# 4. Phân tích kết quả
cluster_profiles = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income ($)': 'mean',
    'Spending Score (1-100)': 'mean',
    'Work Experience': 'mean',
    'Family Size': 'mean',
    'Gender': lambda x: x.mode()[0]
})
print("\nThông tin trung bình của từng cụm:")
print(cluster_profiles)

# Đánh giá chất lượng phân cụm
silhouette = silhouette_score(X_scaled, df['Cluster'])
ch_score = calinski_harabasz_score(X_scaled, df['Cluster'])
db_score = davies_bouldin_score(X_scaled, df['Cluster'])

print(f"\nSilhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Index: {ch_score:.3f}")
print(f"Davies-Bouldin Index: {db_score:.3f}")

# 5. Trực quan hóa kết quả
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Annual Income ($)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', style='Cluster', s=100)
plt.title('Phân cụm khách hàng: Thu nhập vs Điểm chi tiêu')
plt.show()

print("\nMột vài khách hàng sau khi được phân cụm:")
print(df.sample(5))