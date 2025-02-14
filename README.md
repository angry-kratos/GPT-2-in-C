# **Naive GPT-2 Implementation in C**

This repository contains a **naive, from-scratch implementation of a GPT-2-style transformer** written in plain **C**. It serves as a starting point for exploring **parallel computing**, **CUDA optimization**, and understanding the inner workings of transformers at a fundamental level.

## **Overview**

The goal of this project is to:
1. **Understand the GPT-2 architecture** by implementing it from scratch in an unoptimized way.
2. **Learn CUDA step-by-step**, starting with simple operations like matrix multiplications and gradually optimizing this implementation using custom CUDA kernels.
3. **Benchmark and document performance improvements** as optimizations are introduced, making this project both a learning resource and a showcase of iterative optimization.

---

## **How to Run**

To run this implementation, you’ll need a few resources prepared:

### **1. Model Parameters**
You’ll need to download the **GPT-2 model parameters in `safetensors` format** from Hugging Face. Use the following link to download the required file:

👉 **[GPT-2 Safetensors File on Hugging Face](https://huggingface.co/gpt2/resolve/main/model.safetensors)**

Place the downloaded `model.safetensors` file in the same directory as the code.

---

### **2. Tokenized Data and Encoding**
This implementation relies on two additional files:
- **`data`**: Tokenized text data.
- **`enc`**: Encoded vocabulary.

Both files are generated using **tiktoken** (a library for tokenization) and the **Tiny Shakespeare dataset**. You can prepare these files by following these steps:

#### a. Install `tiktoken`
```bash
pip install tiktoken
```

#### b. Download the Tiny Shakespeare dataset
Download the dataset from [Tiny Shakespeare Dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) and save it as `input.txt`.

#### c. Generate `data` and `enc`
Run the following Python script to generate the necessary files:
```python
import tiktoken

# Load Tiny Shakespeare dataset
with open("input.txt", "r") as f:
    text = f.read()

# Initialize tokenizer (GPT-2)
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)

# Save tokenized data
with open("data", "wb") as f:
    f.write(bytes(tokens))

# Save the encoding data
with open("enc", "wb") as f:
    for token, size in enumerate(enc.mergeable_ranks):
        offset = len(f"{token}\n")
        f.write(f"{offset} {size}\n".encode("utf-8"))
```

---

### **3. Compile and Run the Code**

After preparing the necessary files (`model.safetensors`, `data`, and `enc`), you can compile the program using `gcc`:

```bash
gcc -std=c11 gpt.c -o gpt -lm
```

Run the program:
```bash
./gpt
```

---

## **References**

This project was heavily inspired by [this YouTube video]([https://youtu.be/d1LNUvkRMEg?si=5LSbx7ME7Hf4IsuB](https://youtu.be/d1LNUvkRMEg?si=j22jHM_gZWcjw6uW)) by **Raff K**, which provides an excellent explanation of implementing GPT-like transformers in C. I highly recommend checking it out for anyone diving into this project.

---

## **Future Work**

1. **CUDA Optimization**:  
   Starting with basic operations (e.g., matrix multiplication), I’ll iteratively replace key sections with optimized CUDA kernels to improve performance.

2. **Worklog Documentation**:  
   I’ll document the journey of understanding CUDA, optimizing this implementation, and benchmarking each step.

3. **Sharing Learnings**:  
   This project aims to bridge the gap between basic CUDA tutorials and production-grade CUDA (e.g., kernels in [llm.c](https://github.com/karpathy/llm.c) by Karpathy).

---

Feel free to clone this repository and experiment with your own optimizations. Contributions, feedback, and suggestions are welcome!

---

## **Disclaimer**
This implementation is intentionally naive for educational purposes. It is neither optimized for performance nor intended for production use.

---

### **License**
MIT License
