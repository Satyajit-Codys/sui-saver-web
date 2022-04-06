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
from apps.tweetml.ml_model import processing_tweets
from apps.tweetml.ml_model import all_tweets, positives


@login_required(login_url="/login/")
def see_tweets(request):
    processing_tweets()
    msg = None

    return render(request, "tweetml/showtweets.html", {"msg": msg, "all_tweets": all_tweets, "positives": positives, "segment": "tweets"})
