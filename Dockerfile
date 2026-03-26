# 1. 換成 CUDA 11.8 的底層映像檔
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# 2. 設定容器內的工作目錄為 /workspace
WORKDIR /workspace

# 3. 複製套件清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 設定預設進入 bash (終端機介面)
CMD ["bash"]