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
from apps.tweetml.ml_model import *
from apps.questionnaire.views import *
from apps.questionnaire.models import Qresponses


@login_required(login_url="/login/")
def index(request):
    total_tweets = len(positive_tweet_trend) + len(negative_tweet_trend)
    try:
        overall_trend_percentage = sum(overall_trend) // total_tweets
    except:
        overall_trend_percentage = "No data"
    try:
        recent_prob = sum(recent_prob_list)//10
    except:
        recent_prob = "No data"
    res = Qresponses.objects.get(id=1)
    sui_percentage = res.percentage

    context = {"segment": "index", "positive_tweet_trend": positive_tweet_trend,
               "negative_tweet_trend": negative_tweet_trend, "overall_trend_percentage": overall_trend_percentage,
               "recent_prob": recent_prob, "sui_percentage": sui_percentage, "total_tweets": total_tweets}

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))
    # return render(request, "", context,  positive_tweet_trend, negative_tweet_trend)


@ login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        # html_template = loader.get_template('home/' + load_template)
        # return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))
