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
Build and install intel real sense
```bash
git clone https://github.com/JetsonHacksNano/installLibrealsense
./buildLibrealsense.sh -v v2.48.0
sudo cp /home/"$USER"/librealsense/wrappers/python/pyrealsense2/__init__.py /usr/local/lib/python3.6/pyrealsense2/
````

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
Copy python wrapper lib file of intel real sense installed earlier to this virtual environment.

Update $path to virtual environment path in below command.
```bash
cp -r /usr/local/lib/python3.6/pyrealsense2 $path/env/lib/python3.6/site-packages/
```
Run app
```bash
cd embross-emvision-embedded-api/app/
python3 main.py
```

## 3.	OS dependencies
### You need to have Jetpack 4.5.1 installed.

