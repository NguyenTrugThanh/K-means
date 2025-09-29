# Ứng dụng Phân Cụm Khách Hàng bằng K-Means

## Giới thiệu
Ứng dụng web đơn giản để dự đoán phân khúc khách hàng dựa trên mô hình K-Means.  
Người dùng nhập thông tin khách hàng (**Gender, Age, Annual Income, Spending Score, Profession, Work Experience, Family Size**) → hệ thống phân loại khách hàng vào **Cluster** và gán **Segment dễ hiểu** ngay lập tức.

---

## Công nghệ và công cụ sử dụng

| Thành phần              | Công nghệ sử dụng   |
|--------------------------|---------------------|
| Ngôn ngữ lập trình       | Python 3            |
| Web framework            | Flask               |
| Machine Learning         | K-Means             |
| Tối ưu tham số           | Optuna              |
| Xử lý dữ liệu            | pandas, scikit-learn|
| Lưu mô hình              | joblib              |
| Trực quan hóa            | matplotlib, seaborn |
| Frontend                 | HTML, CSS, Bootstrap 5 |

---

## Logic & hoạt động

### 1. Tiền xử lý dữ liệu
- Encode các cột **categorical** (`Gender`, `Profession`) sang số.
- Chuẩn hóa dữ liệu numeric bằng **StandardScaler**.

### 2. Huấn luyện mô hình
- Sử dụng **K-Means** với các feature: `Gender, Age, Annual Income, Spending Score, Profession, Work Experience, Family Size`.
- Xác định số **cluster tối ưu** bằng **Elbow Method**.

### 3. Gán nhãn Segment dễ hiểu cho mỗi cluster
- `Cluster 0`: High Income - High Spending (VIP)  
- `Cluster 1`: High Income - Low Spending  
- `Cluster 2`: Low Income - High Spending  
- `Cluster 3`: Low Income - Low Spending  
- `Cluster 4`: Middle Income - Moderate Spending  

### 4. Lưu mô hình
- Lưu **K-Means**, **scaler**, **encoders**, **cluster labels** ra file `.pkl`.

### 5. Triển khai Flask app
- Giao diện web gồm form nhập dữ liệu khách hàng.  
- Khi submit → Flask backend dự đoán cluster bằng K-Means.  
- Kết quả hiển thị: **Cluster số + Segment dễ hiểu**.  
- Dữ liệu vừa nhập sẽ được giữ nguyên sau khi dự đoán.  

---

## Giao diện
<img width="1241" height="538" alt="image" src="https://github.com/user-attachments/assets/14d5d2e5-37b5-4fe5-a36d-0af1c83f842a" />


<img width="1247" height="549" alt="image" src="https://github.com/user-attachments/assets/92b37241-3b5f-4d6c-bace-ce306270eb9c" />


<img width="1234" height="410" alt="image" src="https://github.com/user-attachments/assets/29edc7f4-578f-4634-9487-aebb2f3990e3" />
