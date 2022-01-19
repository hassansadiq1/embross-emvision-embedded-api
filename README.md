# Introduction 
# embross-emvision-embedded-api

## 1.	Dependencies Installation
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-setuptools libjpeg-turbo8 libjpeg8-dev zlib1g-dev python3-venv python3-bcrypt
```
Adding cuda path
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc 
source ~/.bashrc
```
## 2 Python VENV
```bash
python3 -m venv ./env --system-site-packages
source ./env/bin/activate
pip install -U pip
```
Copy paravision python wheel files and correct path of these files in requirements.txt
```bash
pip install -r requirements.txt
```
Run app
```bash
cd embross-emvision-embedded-api/app/
python3 main.py
```

## 3.	OS dependencies
### You need to have Jetpack 4.5.1 installed.

