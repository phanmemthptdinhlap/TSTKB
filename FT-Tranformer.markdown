### Key Points
- Nghiên cứu cho thấy FT-Transformer là một mô hình mạnh mẽ cho dữ liệu bảng, xử lý cả đặc trưng số và danh mục.
- Mô hình sử dụng nhúng tuyến tính cho đặc trưng số và bảng nhúng cho đặc trưng danh mục, sau đó áp dụng Transformer để học mối quan hệ.
- Có thể cần điều chỉnh siêu tham số để tối ưu hiệu suất, và hiệu quả phụ thuộc vào tập dữ liệu.

### Kiến trúc FT-Transformer
FT-Transformer là một phiên bản của kiến trúc Transformer được thiết kế để xử lý dữ liệu bảng, bao gồm cả đặc trưng số (như tuổi, thu nhập) và đặc trưng danh mục (như giới tính, nghề nghiệp). Mô hình hoạt động như sau:

- **Bộ mã hóa đặc trưng:** Mỗi đặc trưng số được chiếu thành vector nhúng bằng tầng tuyến tính riêng, trong khi đặc trưng danh mục được nhúng bằng bảng nhúng.
- **Kết hợp và thêm mã [CLS]:** Các vector nhúng được kết hợp, thêm mã [CLS] để tổng hợp thông tin.
- **Lớp Transformer:** Chuỗi nhúng đi qua nhiều lớp Transformer để học mối quan hệ giữa các đặc trưng.
- **Dự đoán:** Vector [CLS] cuối cùng được dùng để dự đoán kết quả.

### Ví dụ sử dụng
Dưới đây là một ví dụ đơn giản để minh họa cách triển khai FT-Transformer bằng Python, với tên biến bằng tiếng Việt để dễ hiểu:

```python
import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(self, số_đặc_trưng_số, danh_mục, kích_thước_nhúng, độ_sâu, số_đầu, kích_thước_mlp, số_lớp):
        super(FTTransformer, self).__init__()
        
        self.chiếu_số = nn.ModuleList([nn.Linear(1, kích_thước_nhúng) for _ in range(số_đặc_trưng_số)])
        self.nhúng_danh_mục = nn.ModuleList([nn.Embedding(số_danh_mục, kích_thước_nhúng) for số_danh_mục in danh_mục])
        self.mã_cls = nn.Parameter(torch.randn(1, 1, kích_thước_nhúng))
        tổng_mã = số_đặc_trưng_số + len(danh_mục) + 1
        self.nhúng_vị_trí = nn.Parameter(torch.randn(1, tổng_mã, kích_thước_nhúng))
        lớp_bộ_biến_đổi = nn.TransformerEncoderLayer(d_model=kích_thước_nhúng, nhead=số_đầu, dim_feedforward=kích_thước_mlp)
        self.bộ_biến_đổi = nn.TransformerEncoder(lớp_bộ_biến_đổi, num_layers=độ_sâu)
        self.đầu_mlp = nn.Sequential(
            nn.LayerNorm(kích_thước_nhúng),
            nn.Linear(kích_thước_nhúng, số_lớp)
        )
    
    def forward(self, x_số, x_danh_mục):
        kích_thước_lô = x_số.size(0)
        nhúng_số = [chiếu(x_số[:, i].unsqueeze(1)) for i, chiếu in enumerate(self.chiếu_số)]
        nhúng_số = torch.stack(nhúng_số, dim=1)
        nhúng_danh_mục = [nhúng(x_danh_mục[:, i]) for i, nhúng in enumerate(self.nhúng_danh_mục)]
        nhúng_danh_mục = torch.stack(nhúng_danh_mục, dim=1)
        mã_cls = self.mã_cls.expand(kích_thước_lô, -1, -1)
        x = torch.cat([mã_cls, nhúng_số, nhúng_danh_mục], dim=1)
        x = x + self.nhúng_vị_trí
        x = self.bộ_biến_đổi(x)
        đầu_ra_cls = x[:, 0, :]
        đầu_ra = self.đầu_mlp(đầu_ra_cls)
        return đầu_ra

# Ví dụ sử dụng
số_đặc_trưng_số = 2
danh_mục = [3, 4]
kích_thước_nhúng = 16
độ_sâu = 2
số_đầu = 4
kích_thước_mlp = 32
số_lớp = 2
model = FTTransformer(số_đặc_trưng_số, danh_mục, kích_thước_nhúng, độ_sâu, số_đầu, kích_thước_mlp, số_lớp)
batch_size = 4
x_số = torch.randn(batch_size, số_đặc_trưng_số)
x_danh_mục = torch.randint(0, 3, (batch_size, 2))
đầu_ra = model(x_số, x_danh_mục)
print(đầu_ra.shape)
```

### Tối ưu hóa
Nghiên cứu cho thấy cần điều chỉnh siêu tham số như độ sâu, kích thước nhúng, và tỷ lệ dropout để tối ưu hiệu suất, tùy thuộc vào tập dữ liệu cụ thể.

---

### Báo cáo chi tiết về FT-Transformer

#### Giới thiệu
FT-Transformer, được giới thiệu trong bài báo "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021), là một phiên bản điều chỉnh của kiến trúc Transformer để xử lý dữ liệu bảng, bao gồm cả đặc trưng số và danh mục. Mô hình này nổi bật nhờ khả năng học các mối quan hệ phức tạp giữa các đặc trưng, thường vượt trội hơn các mô hình như gradient boosting và ResNet trên nhiều tập dữ liệu.

#### Kiến trúc chi tiết
Dựa trên tài liệu từ bài báo và các triển khai thực tế, kiến trúc FT-Transformer bao gồm các thành phần chính:

- **Bộ mã hóa đặc trưng (Feature Tokenizer):**
  - Đối với đặc trưng số, mỗi đặc trưng \( x_j \) được chiếu thành vector nhúng \( T_j = b_j + x_j \cdot W_j \), với \( b_j \) là vector bias và \( W_j \in \mathbb{R}^d \) là vector trọng số. Điều này cho phép mỗi đặc trưng số được biểu diễn trong không gian nhúng có kích thước \( d \).
  - Đối với đặc trưng danh mục, mỗi giá trị danh mục được ánh xạ qua một bảng nhúng \( W_j \in \mathbb{R}^{S_j \times d} \), với \( S_j \) là số giá trị có thể của đặc trưng đó.

- **Kết hợp và thêm mã [CLS]:**
  - Tất cả các vector nhúng được kết hợp thành một chuỗi \( T \in \mathbb{R}^{k \times d} \), với \( k \) là tổng số đặc trưng (số và danh mục).
  - Một mã [CLS] đặc biệt được thêm vào đầu chuỗi, giúp tổng hợp thông tin cho dự đoán cuối cùng.

- **Lớp Transformer:**
  - Chuỗi nhúng, bao gồm mã [CLS], được truyền qua \( L \) lớp Transformer, mỗi lớp gồm:
    - Multi-Head Self-Attention (MHSA) để học mối quan hệ giữa các đặc trưng.
    - Feed Forward Network (FFN) để xử lý độc lập từng đặc trưng.
    - Layer Normalization và kết nối dư (residual connections) để ổn định huấn luyện.
  - Sử dụng PreNorm để tối ưu hóa, với việc loại bỏ chuẩn hóa lớp đầu tiên trong tầng đầu tiên để cải thiện hiệu suất.

- **Dự đoán:**
  - Sau khi qua các lớp Transformer, vector nhúng của mã [CLS] được lấy ra và truyền qua một mạng nơ-ron đơn giản: LayerNorm -> ReLU -> Linear, để tạo ra dự đoán cuối cùng.

#### Ví dụ minh họa
Dưới đây là một ví dụ mã giả bằng Python với PyTorch, sử dụng tên biến tiếng Việt để dễ hiểu:

```python
import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    def __init__(self, số_đặc_trưng_số, danh_mục, kích_thước_nhúng, độ_sâu, số_đầu, kích_thước_mlp, số_lớp):
        super(FTTransformer, self).__init__()
        
        self.chiếu_số = nn.ModuleList([nn.Linear(1, kích_thước_nhúng) for _ in range(số_đặc_trưng_số)])
        self.nhúng_danh_mục = nn.ModuleList([nn.Embedding(số_danh_mục, kích_thước_nhúng) for số_danh_mục in danh_mục])
        self.mã_cls = nn.Parameter(torch.randn(1, 1, kích_thước_nhúng))
        tổng_mã = số_đặc_trưng_số + len(danh_mục) + 1
        self.nhúng_vị_trí = nn.Parameter(torch.randn(1, tổng_mã, kích_thước_nhúng))
        lớp_bộ_biến_đổi = nn.TransformerEncoderLayer(d_model=kích_thước_nhúng, nhead=số_đầu, dim_feedforward=kích_thước_mlp)
        self.bộ_biến_đổi = nn.TransformerEncoder(lớp_bộ_biến_đổi, num_layers=độ_sâu)
        self.đầu_mlp = nn.Sequential(
            nn.LayerNorm(kích_thước_nhúng),
            nn.Linear(kích_thước_nhúng, số_lớp)
        )
    
    def forward(self, x_số, x_danh_mục):
        kích_thước_lô = x_số.size(0)
        nhúng_số = [chiếu(x_số[:, i].unsqueeze(1)) for i, chiếu in enumerate(self.chiếu_số)]
        nhúng_số = torch.stack(nhúng_số, dim=1)
        nhúng_danh_mục = [nhúng(x_danh_mục[:, i]) for i, nhúng in enumerate(self.nhúng_danh_mục)]
        nhúng_danh_mục = torch.stack(nhúng_danh_mục, dim=1)
        mã_cls = self.mã_cls.expand(kích_thước_lô, -1, -1)
        x = torch.cat([mã_cls, nhúng_số, nhúng_danh_mục], dim=1)
        x = x + self.nhúng_vị_trí
        x = self.bộ_biến_đổi(x)
        đầu_ra_cls = x[:, 0, :]
        đầu_ra = self.đầu_mlp(đầu_ra_cls)
        return đầu_ra

# Ví dụ sử dụng
số_đặc_trưng_số = 2
danh_mục = [3, 4]
kích_thước_nhúng = 16
độ_sâu = 2
số_đầu = 4
kích_thước_mlp = 32
số_lớp = 2
model = FTTransformer(số_đặc_trưng_số, danh_mục, kích_thước_nhúng, độ_sâu, số_đầu, kích_thước_mlp, số_lớp)
batch_size = 4
x_số = torch.randn(batch_size, số_đặc_trưng_số)
x_danh_mục = torch.randint(0, 3, (batch_size, 2))
đầu_ra = model(x_số, x_danh_mục)
print(đầu_ra.shape)
```

#### Tối ưu hóa và hiệu suất
Nghiên cứu cho thấy FT-Transformer có hiệu suất tốt trên nhiều tập dữ liệu, với xếp hạng trung bình 1.8 so với các mô hình khác (theo bảng 2 trong bài báo). Tuy nhiên, mô hình yêu cầu nhiều tài nguyên hơn so với các mô hình đơn giản như ResNet, với thời gian huấn luyện lâu hơn (ví dụ: 13.8 lần trên tập dữ liệu Yahoo, bảng 10). Độ phức tạp bậc hai của MHSA có thể được giảm bằng cách sử dụng các phương pháp xấp xỉ hiệu quả.

#### So sánh với các mô hình khác
Dưới đây là bảng so sánh FT-Transformer với các mô hình khác dựa trên tài liệu:

| Mô hình            | Ưu điểm                              | Nhược điểm                          |
|-------------------|--------------------------------------|-------------------------------------|
| FT-Transformer    | Xử lý tốt cả số và danh mục, hiệu suất cao | Yêu cầu tài nguyên lớn, huấn luyện lâu |
| Gradient Boosting | Nhanh, hiệu quả trên nhiều tập dữ liệu | Không học được mối quan hệ phức tạp |
| ResNet            | Đơn giản, ít tài nguyên             | Hiệu suất thấp hơn trên nhiều nhiệm vụ |

#### Tài nguyên và triển khai
- Gói Python để triển khai: `tabtransformertf`, cài đặt bằng `pip install tabtransformertf`.
- Mã nguồn: Có sẵn tại [GitHub](https://github.com/yandex-research/tabular-dl-revisiting-models).
- Ví dụ thực tế: Đánh giá trên tập dữ liệu Adult Income Dataset ([tập dữ liệu](https://archive.ics.uci.edu/ml/datasets/adult)), đạt độ chính xác 0.8576, gần với báo cáo 0.86.

#### Kết luận
FT-Transformer là một công cụ mạnh mẽ cho dữ liệu bảng, với khả năng xử lý cả đặc trưng số và danh mục. Tuy nhiên, cần tối ưu hóa siêu tham số và cân nhắc tài nguyên khi triển khai. Ví dụ mã trên minh họa cách triển khai cơ bản, và người dùng có thể điều chỉnh theo nhu cầu cụ thể.

### Key Citations
- [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/pdf/2106.11959)
- [Adult Income Dataset for Machine Learning](https://archive.ics.uci.edu/ml/datasets/adult)
- [GitHub Repository for FT-Transformer](https://github.com/yandex-research/tabular-dl-revisiting-models)