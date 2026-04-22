# Banking Intent Classification với Unsloth

Đồ án fine-tune model **Llama-3.2-3B-Instruct** để phân loại ý định (intent) của khách hàng ngân hàng, sử dụng thư viện [Unsloth](https://unsloth.ai/) trên tập dữ liệu [BANKING77](https://huggingface.co/datasets/PolyAI/banking77).

## Tổng quan

Ý tưởng chính của project là dùng một mô hình ngôn ngữ lớn (LLM) để làm bài toán phân loại intent. Thay vì dùng classification head truyền thống, tôi sử dụng cách tiếp cận **generative classification** — tức là train model sinh ra tên intent dưới dạng text.

Model được fine-tune bằng phương pháp **QLoRA** (Quantized LoRA) thông qua Unsloth SFTTrainer, chạy trên Google Colab Free (GPU T4).

**Thông tin chính:**
- **Model gốc:** `unsloth/Llama-3.2-3B-Instruct`
- **Dataset:** BANKING77 (lấy 45 intents, khoảng 7,600 mẫu)
- **Phương pháp:** QLoRA 4-bit + SFTTrainer (Unsloth)
- **Môi trường:** Google Colab Free Tier, T4 GPU

## Video Demo

> **[📹 Xem video demo trên Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)**

## Cấu trúc thư mục

```
banking-intent-unsloth/
├── scripts/
│   ├── train.py             
│   ├── inference.py          
│   └── preprocess_data.py    
├── configs/
│   ├── train.yaml           
│   ├── inference.yaml        
├── sample_data/
│   ├── train.csv           
│   └── test.csv              
├── notebooks/
│   └── train_colab_final.ipynb 
├── train.sh                  
├── inference.sh              
├── requirements.txt
└── README.md
```

## Hướng dẫn chạy

### Cách 1: Google Colab (khuyến nghị)

1. Mở notebook trên Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

2. Chọn Runtime → Change runtime type → **T4 GPU**
3. Chạy lần lượt từng cell, notebook sẽ tự cài đặt thư viện, train và đánh giá model

### Cách 2: Chạy local bằng CLI

#### Cài đặt

```bash
git clone https://github.com/YOUR_USERNAME/banking-intent-unsloth.git
cd banking-intent-unsloth
pip install -r requirements.txt
```

#### Tiền xử lý dữ liệu

```bash
python scripts/preprocess_data.py --config configs/train.yaml
```

Lệnh này sẽ tải dataset BANKING77 từ HuggingFace, chọn ngẫu nhiên 45 intents, format thành dạng instruction-output rồi chia train/test.

#### Training

```bash
python scripts/train.py --config configs/train.yaml
# hoặc
bash train.sh
```

#### Inference

Phân loại 1 câu:
```bash
python scripts/inference.py --config configs/inference.yaml --message "I am still waiting on my card?"
```

Chế độ tương tác (gõ câu hỏi liên tục):
```bash
bash inference.sh
```

#### Sử dụng trong Python

```python
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")
label = classifier("I am still waiting on my card?")
print(label)  # → card_arrival
```

## Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| Model gốc | `unsloth/Llama-3.2-3B-Instruct` |
| Quantization | 4-bit (QLoRA) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.0 |
| Target Modules | q, k, v, o, gate, up, down proj |
| Batch Size | 8 |
| Gradient Accumulation | 4 (effective batch = 32) |
| Learning Rate | 2e-4 |
| Optimizer | AdamW 8-bit |
| LR Scheduler | Cosine |
| Số epochs | 4 |
| Max Sequence Length | 256 |
| Weight Decay | 0.01 |
| Warmup Steps | 15 |

## Kết quả

| Chỉ số | Điểm |
|--------|------|
| **Test Accuracy** | 0.95 |
| **Macro F1** | 0.85 |

## Về Dataset

- **Nguồn:** [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- **Gốc:** 77 intents, 13,083 mẫu
- **Sử dụng:** 45 intents (chọn ngẫu nhiên), khoảng 7,600 mẫu
- **Chia tập:** 80% train / 20% test (stratified theo label)

## Tài liệu tham khảo

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [BANKING77 Paper](https://arxiv.org/abs/2003.04807)
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## Ghi chú

Đồ án này được thực hiện cho môn học NLP Doanh Nghiệp, Trường ĐH Khoa học Tự nhiên — ĐHQG TP.HCM.  
Dataset BANKING77 sử dụng giấy phép [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
