FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get -y install gcc g++
RUN conda install scipy==1.11.1 numpy==1.24.3 matplotlib==3.7.1 numba==0.57.0
RUN conda install pytorch==2.0.1 torchvision==0.15.2 -c pytorch
RUN pip install scikit-learn==1.3.0 tqdm==4.65.0 tensorflow==2.13.0 keras==2.13.1 tensorboardX==2.6.2.2 xgboost==1.7.6 lightgbm==4.0.0
RUN pip install pathfinding==1.0.4
RUN conda install pandas==2.0.3
RUN pip install networkx==3.1 dill==0.3.7 pyastar2d==1.0.6 easydict==1.10 sympy==1.11.1 pygame==2.5.1
COPY . .
CMD /bin/bash
