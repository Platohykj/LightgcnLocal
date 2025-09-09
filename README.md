LightGCN

本项目是对[LightGCNpp](https://github.com/geon0325/LightGCNpp) 项目的整理

目前只包括源仓库的训练部分代码

本仓库经验证可以在 Windows 上采用以下流程进行环境配置

### 环境配置
#### 1.创建conda虚拟环境
```powershell
conda create -n lightgcn python=3.10 -y
conda activate lightgcn
```

#### 2.安装pytorch
应根据自己的硬件环境选择合适的安装命令
参考 [pytorch官网](https://pytorch.org/get-started/locally/)
在我的环境中，安装命令为
```powershell
pip3 install torch --index-url https://download.pytorch.org/whl/cu129
```
#### 3.安装其他依赖
```powershell
pip install pandas
pip install scipy
pip install scikit-learn
pip install tqdm
pip install tensorboardx
```