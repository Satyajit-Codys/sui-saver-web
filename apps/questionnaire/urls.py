from django.urls import path, re_path
from apps.questionnaire import views

urlpatterns = [

    # The tweets page
    path('dashboard-questions/', views.see_questions, name='see_questions'),

]
