# 1. 換成 CUDA 11.8 的底層映像檔
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# 2. 設定容器內的工作目錄為 /workspace
WORKDIR /workspace

# 3. 複製套件清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install jupyterlab

# 5. 告訴系統預設會使用 8888 通訊埠
EXPOSE 8888

# 6. 容器啟動時改為執行 Jupyter (取代原本的 bash)，並設定免密碼登入
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]