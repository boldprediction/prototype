#!/bin/bash
rm -rf ~/Library/Application Support/pycortex/
virtualenv -p /usr/bin/python2.7 venv
source venv/bin/activate
pip install django
pip install django-hashid-field==2.1.6
pip install mysqlclient==1.4.2
pip install celery==4.3.0
pip install numpy==1.16.2
pip install tables==3.5.1
pip install nibabel==2.4.0
pip install Cython==0.29.6
pip install nipy==0.4.2
pip install h5py==2.9.0
pip install Pillow==6.0.0
pip install seaborn==0.9.0
pip install cortex/
deactivate
