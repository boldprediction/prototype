from __future__ import unicode_literals
import datetime

from django.db import models
from django.utils import timezone
import json
jsonDec = json.decoder.JSONDecoder()
from hashid_field import HashidAutoField



# Create your models here.

#from django.db import models
from django import forms


class Contrast(models.Model):

    id = HashidAutoField(primary_key=True)
    list1_name = models.TextField('Enter name of Condition 1')
    list1_text = models.TextField('Enter stimulus words separated by a comma')
    baseline_choice = models.BooleanField('Compare to baseline.', default=False)
    list2_name = models.TextField('Enter name of Condition 2')
    list2_text = models.TextField ('Enter stimulus words separated by a comma')
    permutation_choice = models.BooleanField('Run permutation / bootstrap test (requires more waiting time).', default=False)
    experiment_id = models.BigIntegerField('experiment if it exists',default=0)
    figures_list = models.TextField('Enter name of Condition 1', default = '')
    MNIstr = models.TextField('model str')
    subjstr = models.TextField('model str')
    pmaps = models.TextField('model str')
    replicated_figure = models.TextField('replicated_image',  default = '')
    random_roi_file = models.TextField('random_roi_file',  default = '')


    def get_str(self):
        strs = dict()
        strs['list1'] = self.list1_text
        strs['list2'] = self.list2_text
        strs['list1_name'] = self.list1_name
        strs['list2_name'] = self.list2_name
        strs['do_perm'] = self.permutation_choice
        return strs
    
    def get_MNI_names(self):
        strs = dict()
        strs['Cstr'] = self.MNIstr
        print strs['Cstr']
        return strs

    def get_subj_names(self, isub):
        strs = dict()
        strs['Cstr'] = jsonDec.decode(self.subjstr)[int(isub)-1]
        print strs['Cstr']
        return strs


class ExpManager(models.Manager):
    def create_exp(self, name, contrast_res):
        e = self(name = name, contrast_res = contrast_res)
        # do something with the book
        return e


class Exp(models.Model):

    name = models.TextField('experiment_name')
    authors = models.TextField('experiment_name')
    DOI = models.TextField('experiment_name')
    title = models.TextField('experiment_name')
    contrasts_res = models.TextField('model str')
    objects = ExpManager()

    # def create(cls, name, contrast_res):
    #     e = cls(name = name, contrast_res = contrast_res)
    #     # do something with the book
    #     return e

#
# class Condition(models.Model):
#     name = models.TextField(max_length=200)
#     word_list = models.TextField(max_length=10000)
#
#

class Coordinates_holder(models.Model):
    title = models.TextField('', default = '')
    roi_image_filename = models.TextField('',  default = '')
    allmasks = models.TextField('',  default = '')
    contrast = models.ForeignKey(Contrast)

class Coordinates(models.Model):
    name = models.TextField('roi name')
    x = models.IntegerField('x')
    y = models.IntegerField('y')
    z = models.IntegerField('z')
    coordinates_holder = models.ForeignKey(Coordinates_holder)

