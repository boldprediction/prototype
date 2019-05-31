from __future__ import absolute_import
import os

from celery import Celery
from celery import Task

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'boldpredictions.settings')
#os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
app = Celery('simulate', backend='rpc://guest@localhost', broker = 'amqp://guest:@localhost')#broker = 'pyamqp://guest@localhost//')#, broker='amqp://localhost//')
app.config_from_object('django.conf:settings')
#app.autodiscover_tasks()
