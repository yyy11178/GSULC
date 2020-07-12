Introduction
---
Datasets and source code for our paper **A Global Selection and Uncertainly Loss Correction
Method for Web-Supervised Fine-Grained Visual Classification**


Network Architecture
---
![network_architecture](image/network_architecture.png)


Installation
---
```
- pytorch, tested on [v1.0]
- CUDA, tested on v9.0
- Language: Python 3.6
```


How to use
---
The code is currently tested only on GPU
- Data Preparation

   Download data into working directory and decompress them using
   ```
   wget https://web-fgvc-496-5089.oss-cn-hongkong.aliyuncs.com/web-aircraft.tar.gz
   wget https://web-fgvc-496-5089.oss-cn-hongkong.aliyuncs.com/web-bird.tar.gz
   wget https://web-fgvc-496-5089.oss-cn-hongkong.aliyuncs.com/web-car.tar.gz
   tar -xvf web-aircraft.tar.gz
   tar -xvf web-bird.tar.gz
   tar -xvf web-car.tar.gz
   ```
   
   
- Source Code
    
    - If you want to train the whole model from beginning using the source code, please follow the subsequent steps.

        - Download dataset of `web-bird`/`web-aircraft`/`web-car` into the working directory as needed.
        - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id in `web_birds.sh`, `web_aircrafts.sh`, `web_cars.sh`.
        - Run the script
            ```
            bash web_birds.sh
            bash web_aircrafts.sh
            bash web_cars.sh
            ```

