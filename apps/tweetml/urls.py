# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.tweetml import views

urlpatterns = [

    # The tweets page
    path('dashboard-tweets/', views.see_tweets, name='see_tweets'),


]
