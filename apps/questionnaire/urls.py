# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django.urls import path, re_path
from apps.questionnaire import views

urlpatterns = [

    # The tweets page
    path('dashboard-questions/', views.see_questions, name='see_questions'),

]
