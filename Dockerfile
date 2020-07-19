FROM centos
RUN yum install epel-release -y
RUN yum install python36 -y
RUN pip3 install --upgrade pip
RUN pip3 install pillow
RUN pip3 install keras
RUN pip3 install tensorflow
RUN pip3 install pandas
CMD [ "python3", "/home/mycode.py" ]
