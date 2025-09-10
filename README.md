# LightGCN

This project is a curated version of [LightGCNpp](https://github.com/geon0325/LightGCNpp).

It currently includes only the training part of the original repository.

This repository has been validated to set up the environment on Windows using the following steps.

### Environment Setup
#### 1. Create a conda virtual environment
```powershell
conda create -n lightgcn python=3.10 -y
conda activate lightgcn
```

#### 2. Install PyTorch
Choose the appropriate install command for your hardware.
See the [official PyTorch site](https://pytorch.org/get-started/locally/).
In my environment, the installation command is:
```powershell
pip3 install torch --index-url https://download.pytorch.org/whl/cu129
```

#### 3. Install other dependencies
```powershell
pip install pandas
pip install scipy
pip install scikit-learn
pip install tqdm
pip install tensorboardx
```

### Run
```shell
cd src
python main.py <args>
```
