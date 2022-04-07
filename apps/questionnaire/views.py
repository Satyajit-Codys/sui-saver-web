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


@login_required(login_url="/login/")
def see_questions(request):
    msg = None
    if request.method == "POST":
        print(request.form.to_dict())

    return render(request, "questionnaire/questions.html", {"msg": msg, "segment": "questions"})
