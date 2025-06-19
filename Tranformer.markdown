## **Tài liệu chi tiết về mô hình Transformer với PyTorch**

### **Mục lục**
1. Giới thiệu về mô hình Transformer  
2. Cấu trúc tổng quan của Transformer  
   - Encoder  
   - Decoder  
3. Phân tích chi tiết các thành phần  
   - Self-Attention và Multi-Head Attention  
   - Positional Encoding  
   - Feed-Forward Neural Network (FFN)  
   - Layer Normalization và Residual Connections  
4. Triển khai Transformer bằng PyTorch  
   - Positional Encoding  
   - Multi-Head Attention  
   - Feed-Forward Network  
   - Encoder Layer  
   - Decoder Layer  
   - Mô hình Transformer hoàn chỉnh  
5. Ví dụ ứng dụng: Dịch máy  
   - Chuẩn bị dữ liệu  
   - Huấn luyện mô hình  
   - Dự đoán với nhiều ví dụ  
6. Kết luận và hướng mở rộng  

---

### **1. Giới thiệu về mô hình Transformer**

Mô hình Transformer, được giới thiệu trong bài báo *"Attention is All You Need"* (Vaswani et al., 2017), là một kiến trúc mạng nơ-ron mang tính cách mạng trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP). Không giống các mô hình truyền thống như RNN hay LSTM, Transformer loại bỏ cấu trúc tuần tự, thay vào đó xử lý toàn bộ chuỗi đầu vào song song nhờ cơ chế **attention**. Điều này không chỉ tăng tốc độ huấn luyện mà còn cải thiện hiệu suất trên các bài toán phức tạp như dịch máy, tóm tắt văn bản, và sinh văn bản.

#### **Đặc điểm nổi bật**
- **Xử lý song song**: Thay vì duyệt qua chuỗi từng bước như RNN, Transformer tính toán toàn bộ chuỗi cùng lúc.
- **Cơ chế attention**: Tập trung vào các phần quan trọng của dữ liệu mà không cần quét tuần tự, giúp nắm bắt mối quan hệ xa trong chuỗi.
- **Ứng dụng đa dạng**: Ngoài NLP, Transformer còn được áp dụng trong thị giác máy tính (Vision Transformer) và xử lý âm thanh.

#### **Ví dụ minh họa**
Hãy tưởng tượng bạn đang dịch câu: "The cat sits on the mat" sang tiếng Việt. Với RNN, mô hình sẽ xử lý từng từ ("The" → "cat" → "sits" → ...), nhưng Transformer sẽ xem xét toàn bộ câu cùng lúc và quyết định mối liên hệ giữa "cat" và "mat" ngay lập tức nhờ attention.

Mục tiêu của tài liệu này là phân tích chi tiết cấu trúc Transformer, giải thích từng thành phần với công thức và ví dụ, đồng thời triển khai một mô hình hoàn chỉnh bằng PyTorch cho bài toán dịch máy.

### **2. Cấu trúc tổng quan của Transformer**

Transformer bao gồm hai thành phần chính: **Encoder** (bộ mã hóa) và **Decoder** (bộ giải mã), mỗi thành phần được xây dựng từ nhiều tầng (layers) giống nhau.

#### **2.1. Encoder**
Encoder nhận chuỗi đầu vào (ví dụ: câu tiếng Anh) và chuyển đổi thành một tập hợp các vector biểu diễn ẩn. Mỗi tầng Encoder bao gồm:
- **Input Embedding**: Chuyển đổi token (từ hoặc subword) thành vector số.
- **Positional Encoding**: Thêm thông tin về vị trí của token trong chuỗi.
- **Multi-Head Self-Attention**: Tính toán mức độ liên quan giữa các token trong chuỗi đầu vào.
- **Feed-Forward Neural Network (FFN)**: Áp dụng biến đổi phi tuyến tính lên từng vector.
- **Layer Normalization và Residual Connections**: Ổn định quá trình huấn luyện.

#### **2.2. Decoder**
Decoder sinh chuỗi đầu ra (ví dụ: câu tiếng Việt) từ biểu diễn của Encoder. Mỗi tầng Decoder bao gồm:
- **Masked Multi-Head Self-Attention**: Ngăn việc chú ý đến các token tương lai trong chuỗi đầu ra.
- **Cross-Attention**: Kết nối với đầu ra của Encoder để "nhìn" vào chuỗi nguồn.
- **Feed-Forward Neural Network (FFN)**: Tương tự Encoder.
- **Layer Normalization và Residual Connections**: Tương tự Encoder.

#### **2.3. Output Layer**
- Một lớp tuyến tính kết hợp với hàm **softmax** để dự đoán token tiếp theo trong chuỗi đầu ra.

#### **Ví dụ minh họa**
- **Đầu vào**: "I love Hanoi" (Encoder xử lý).
- **Đầu ra**: "Tôi yêu Hà Nội" (Decoder sinh ra từng từ một, dựa trên Encoder).

### **3. Phân tích chi tiết các thành phần**

#### **3.1. Self-Attention và Multi-Head Attention**

**Self-Attention** là cơ chế cốt lõi của Transformer, cho phép mô hình "chú ý" đến các phần khác nhau của chuỗi đầu vào. Công thức tính Self-Attention là:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

- **Thành phần**:
  - \(Q\) (Query): Vector truy vấn, đại diện cho token hiện tại.
  - \(K\) (Key): Vector khóa, đại diện cho các token để so sánh.
  - \(V\) (Value): Vector giá trị, chứa thông tin thực sự của token.
  - \(d_k\): Kích thước của vector Key, dùng để chuẩn hóa nhằm tránh giá trị quá lớn.

- **Quá trình**:
  1. Tính tích vô hướng \(QK^T\) để đo độ tương đồng giữa Query và Key.
  2. Chia cho \(\sqrt{d_k}\) để ổn định gradient.
  3. Áp dụng \(\text{softmax}\) để chuyển thành trọng số attention.
  4. Nhân với \(V\) để tạo vector đầu ra.

- **Ví dụ minh họa**:
  Trong câu "Tôi yêu Hà Nội", khi xử lý từ "yêu":
  - \(Q\): Vector của "yêu".
  - \(K\): Vector của "Tôi", "yêu", "Hà", "Nội".
  - \(V\): Thông tin từ các từ này.
  - Kết quả: Attention score cao cho "Tôi" (chủ ngữ) và "Hà Nội" (tân ngữ), thấp cho "yêu" (chính nó).

**Multi-Head Attention** mở rộng Self-Attention bằng cách:
- Chia không gian embedding thành nhiều "head" (ví dụ: 8 head).
- Tính attention song song trên từng head.
- Ghép kết quả lại để tạo biểu diễn đa dạng hơn.

- **Ví dụ minh họa**:
  - Head 1: Tập trung vào mối quan hệ ngữ pháp (chủ ngữ - động từ).
  - Head 2: Tập trung vào ngữ nghĩa (động từ - tân ngữ).
  - Kết quả: Một biểu diễn phong phú hơn cho từ "yêu".

#### **3.2. Positional Encoding**

Vì Transformer không có cấu trúc tuần tự, nó cần **Positional Encoding** để cung cấp thông tin về vị trí của token. Công thức là:

\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]
\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

- **Ý nghĩa**:
  - \(pos\): Vị trí của token (0, 1, 2, ...).
  - \(i\): Chiều trong vector embedding.
  - \(d_{model}\): Kích thước embedding (thường là 512).

- **Ví dụ minh họa**:
  Với \(pos = 1\), \(i = 0\), \(d_{model} = 512\):
  - \(PE_{(1, 0)} = \sin(1 / 10000^{0}) = \sin(1) \approx 0.841\)
  - \(PE_{(1, 1)} = \cos(1 / 10000^{0}) = \cos(1) \approx 0.540\)
  Vector này được cộng vào embedding của token ở vị trí 1.

#### **3.3. Feed-Forward Neural Network (FFN)**

FFN là một mạng nơ-ron fully-connected áp dụng trên từng vector:

\[
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]

- **Ý nghĩa**: Thêm khả năng học các biểu diễn phi tuyến tính.
- **Ví dụ minh họa**:
  Vector của từ "yêu" sau attention được đưa vào FFN để tạo ra một biểu diễn mới, nhấn mạnh ngữ nghĩa tình cảm.

#### **3.4. Layer Normalization và Residual Connections**

- **Residual Connections**: Thêm đầu vào vào đầu ra của mỗi tầng con:
  \[
  x' = x + \text{Self-Attention}(x)
  \]
  Giúp gradient chảy tốt hơn trong huấn luyện.

- **Layer Normalization**: Chuẩn hóa từng vector:
  \[
  \text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
  \]
  Ổn định giá trị đầu ra.

- **Ví dụ minh họa**:
  Sau Self-Attention, vector của "yêu" được cộng với chính nó (residual) và chuẩn hóa để tránh giá trị quá lớn.

### **4. Triển khai Transformer bằng PyTorch**

Dưới đây là mã nguồn chi tiết để xây dựng Transformer bằng PyTorch.

#### **4.1. Positional Encoding**

```python
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
```

- **Giải thích**: Tạo ma trận positional encoding và cộng vào embedding.

#### **4.2. Multi-Head Attention**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linears[-1](context)
```

- **Giải thích**: Tính attention trên nhiều head, hỗ trợ mask để che các token không liên quan.

#### **4.3. Feed-Forward Network**

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))
```

#### **4.4. Encoder Layer**

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x
```

#### **4.5. Decoder Layer**

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.layernorm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.layernorm2(x + self.dropout(cross_attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout(ffn_output))
        return x
```

#### **4.6. Mô hình Transformer hoàn chỉnh**

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
```

### **5. Ví dụ ứng dụng: Dịch máy**

#### **5.1. Chuẩn bị dữ liệu**
- **Tập dữ liệu**: Sử dụng IWSLT (dịch tiếng Anh - tiếng Việt).
- **Tiền xử lý**:
  - Token hóa: Chia câu thành các từ (hoặc subword với BPE).
  - Xây dựng từ điển: Gán số cho mỗi token (ví dụ: "I" → 5, "love" → 10).
  - Padding: Đệm chuỗi ngắn bằng 0 để bằng độ dài tối đa.

- **Ví dụ**:
  - Câu nguồn: "I love Hanoi" → `[5, 10, 15, 0]`
  - Câu đích: "Tôi yêu Hà Nội" → `[8, 12, 20, 25]`

#### **5.2. Huấn luyện mô hình**

```python
# Khởi tạo mô hình
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_len = 100

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)

# Hàm mất mát và tối ưu hóa
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Huấn luyện (giả sử có train_loader)
for epoch in range(10):
    transformer.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(train_loader):
        optimizer.zero_grad()
        output = transformer(src, tgt[:, :-1])
        loss = criterion(output.view(-1, tgt_vocab_size), tgt[:, 1:].view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
```

#### **5.3. Dự đoán với nhiều ví dụ**

```python
def translate(transformer, src, max_len=100):
    transformer.eval()
    with torch.no_grad():
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        enc_output = transformer.encoder_embedding(src)
        enc_output = transformer.positional_encoding(enc_output)
        for layer in transformer.encoder_layers:
            enc_output = layer(enc_output, src_mask)

        tgt = torch.tensor([[1]])  # <sos>
        for _ in range(max_len):
            tgt_mask = transformer.generate_mask(src, tgt)[1]
            dec_output = transformer.decoder_embedding(tgt)
            dec_output = transformer.positional_encoding(dec_output)
            for layer in transformer.decoder_layers:
                dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
            output = transformer.fc(dec_output[:, -1, :])
            next_token = output.argmax(dim=-1).unsqueeze(0)
            tgt = torch.cat((tgt, next_token), dim=1)
            if next_token.item() == 2:  # <eos>
                break
    return tgt

# Ví dụ dự đoán
examples = [
    torch.tensor([[5, 10, 15, 0]]),  # "I love Hanoi"
    torch.tensor([[7, 12, 0, 0]]),   # "She runs"
    torch.tensor([[9, 14, 16, 0]])   # "He eats rice"
]
for src in examples:
    translation = translate(transformer, src)
    print(f"Input: {src}, Output: {translation}")
```

- **Kết quả giả định**:
  - "I love Hanoi" → "Tôi yêu Hà Nội"
  - "She runs" → "Cô ấy chạy"
  - "He eats rice" → "Anh ấy ăn cơm"

### **6. Kết luận và hướng mở rộng**

Mô hình Transformer là một bước tiến lớn trong xử lý chuỗi nhờ khả năng xử lý song song và cơ chế attention linh hoạt. Tài liệu này đã:
- Phân tích chi tiết cấu trúc và các thành phần của Transformer.
- Triển khai mô hình bằng PyTorch.
- Áp dụng vào bài toán dịch máy với nhiều ví dụ minh họa.