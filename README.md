# QLoRA Fine-Tuning of LLaMA-2-7B on the Finance-Alpaca Dataset

This project fine-tunes **LLaMA-2-7B** using **QLoRA** (4-bit quantization + LoRA adapters) on the **Finance-Alpaca** dataset.  
The goal is to efficiently adapt a large language model to a **financial question-answering** domain using modern, resource-efficient techniques.

---

## Project Overview

Large models like LLaMA-2 normally require significant GPU memory to fine-tune.  
This project uses **QLoRA**, which combines:

- **4-bit quantization (BitsAndBytes)** → drastically reduces memory usage  
- **LoRA adapters (PEFT)** → trains only low-rank matrices in attention layers  
- **TRL SFTTrainer** → streamlined supervised fine-tuning  

This enables full fine-tuning of a 7B model on a **single consumer GPU / Colab instance**.

The model was trained on:
- **1,000 training examples**
- **300 evaluation examples**

from the **gbharti/finance-alpaca** dataset.

---

## Fine-Tuning Pipeline

### **1. Data Preparation**
- Load the Finance-Alpaca dataset (Hugging Face)
- Sample 1000 train / 300 test examples
- Convert each example into Alpaca-style prompts:


### **2. Baseline Evaluation**
Evaluate the pretrained LLaMA-2-7B using ROUGE metrics:
- ROUGE-1  
- ROUGE-2  
- ROUGE-L  
- ROUGE-Lsum  

### **3. QLoRA Fine-Tuning**
- Load LLaMA-2-7B in **4-bit** mode using `BitsAndBytesConfig`
- Add LoRA adapters to:
- `q_proj`
- `v_proj`
- Train using `SFTTrainer` (1 epoch, efficient batch configuration)
- Save adapter weights

### **4. Post-Training Evaluation**
Reload the model with LoRA adapters and re-run ROUGE evaluation on the 300-sample test set.

---

## Results (ROUGE Scores)

| Metric       | Pretrained | Fine-Tuned |
|--------------|------------|------------|
| **ROUGE-1**  | 0.1205     | **0.2519** |
| **ROUGE-2**  | 0.0203     | **0.0400** |
| **ROUGE-L**  | 0.0802     | **0.1371** |
| **ROUGE-Lsum** | 0.0818   | **0.1526** |

**All evaluation metrics improved significantly** after domain-specific fine-tuning.

---

## Tools & Libraries

- **Hugging Face Transformers**
- **PEFT (LoRA Adapters)**
- **TRL (SFTTrainer)**
- **BitsAndBytes (4-bit quantization)**
- **evaluate / ROUGE**
- **PyTorch**

---

## How to Run

- Must be ran using T4 GPU
Install dependencies:

```bash
pip install transformers peft trl bitsandbytes datasets evaluate rouge_score



