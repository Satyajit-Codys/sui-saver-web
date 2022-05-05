# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.db import models

# Create your models here.


class Qresponses(models.Model):
    percentage = models.FloatField()

    class Meta:
        db_table = "qresponses"
