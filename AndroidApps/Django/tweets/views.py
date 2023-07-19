from django.shortcuts import render
from .models import Tweet
from django.http import HttpResponse, Http404, JsonResponse
from .forms import TweetForm

def home_view(request,*args ,**kwargs):

    return render(request,"pages/home.html",context={},status=200)

# def tweet_detail_view(request,tweet_id,*args,**kwargs):
#     return

def tweet_create_view_pure_django(request,*args ,**kwargs):
    form = TweetForm(request.POST or None)
    if form.is_valid():
        obj = form.save(commit=False)
        obj.save()
        form=TweetForm()
    return render(request,'components/9_form.html',context={"form":form})

def tweet_list_view(request,*args,**kwargs):
    qs = Tweet.objects.all()
    tweets_list = [{"id":x.id , "content":x.content}for x in qs]
    data = {
        "response" : tweets_list
    }
    return JsonResponse(data)