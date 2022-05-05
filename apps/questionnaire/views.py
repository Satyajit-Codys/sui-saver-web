# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from django.shortcuts import render, redirect
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
from apps.questionnaire.models import Qresponses


@login_required(login_url="/login/")
def see_questions(request):
    global sui_percentage
    msg = None
    if request.method == "POST":
        form_data = request.POST.dict()
        form_values = list(form_data.values())[5:42]
        print(form_values)
        total_form_value = 0
        for i in form_values:
            total_form_value += int(i)
        sui_percentage = (total_form_value/166)*100
        print(sui_percentage)
        s = Qresponses()
        s.percentage = sui_percentage
        s.save()

    return render(request, "questionnaire/questions.html", {"msg": msg, "segment": "questions"})
