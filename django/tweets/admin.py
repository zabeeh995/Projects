from django.contrib import admin

from .models import Tweet

class TweetAdmin(admin.ModelAdmin):
    search_fields = ['content','user_name','user_email']
    class Meta:
        model = Tweet

admin.site.register(Tweet,TweetAdmin)
