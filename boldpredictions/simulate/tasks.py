#from celery import Celery

#import sys
#sys.path.append('/auto/k1/leila/code/replication/code')
#from utils import read_json
#from experiment import Experiment
#from subject import Subject
#from subject_analyses import SubjectAnalysis
#from subject_group import SubjectGroup
#from result import Result
#import cortex
# import numpy as np
from .replicate import Replicate

#app = Celery('tasks', backend='rpc://', broker='amqp://guest@localhost//')


from .celery import app as celery


@celery.task(base = Replicate)#, broker = 'pyamqp://guest@localhost//')
def make_contrast(inputs, type = '1'):
    if type == '1':
        return make_contrast.compute(inputs)
    if type == '2':
        return make_contrast.run_ROI_table(**inputs)


"""
info = {'DOI': '',
 'contrasts': {'contrast1': {'condition1': ['cond1'],
   'condition2': ['cond2'],
   'coordinates': [],
   'figures': []}},
 'coordinate_space': 'mni',
 'stimuli': {'cond1': {'type': 'word_list',
   'value': 'apple, tomato, meat, sugar'},
  'cond2': {'type': 'word_list', 'value': 'house, car, boat, elevator'}}}
"""
