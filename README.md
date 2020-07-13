Introduction
---
Datasets and source code for our paper **A Global Selection and Uncertainly Loss Correction
Method for Webly Supervised Fine-Grained Visual
Classification**


Network Architecture
---
![network_architecture](image/network_architecture.png)


Installation
---
The code is currently tested in follow virtual environment
- Python 3.7.3

- Pytorch 1.3.1

- CUDA 10.0.130



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
        - Change parameters in `web_birds.sh`, `web_aircrafts.sh`, `web_cars.sh` if you need
            ```
            -CUDA_VISIBLE_DEVICES: gpu number which you want to use
            -dataset: dataset in {web-bird, web-aircraft, web-car}
            -n_classes: classes for different dataset
            -base_lr: initial learning rate
            -batch_size: the proper batch size
            -epoch: the training epoch of every step
            -drop_rate: the rate which you want to corrupt (for different dataset)
            -queue_size: the length of prediction record history
            -warm_up: warm-up training in the begining of training
            ```        
        - Run the script
            ```
            bash web_birds.sh
            bash web_aircrafts.sh
            bash web_cars.sh
            ```
        - The result will be saved as `training_log.txt`

