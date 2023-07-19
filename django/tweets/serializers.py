from rest_framework import serializers
from .models import Tweet
from django.conf import settings

MAX_TWEET_LENGTH = settings.MAX_TWEET_LENGTH

class TweetSerializers(serializers.ModelSerializer):
    class Meta:
        model = Tweet
        fields =['content']

    def validate_content(self,value):
        if len(value) > MAX_TWEET_LENGTH:
            raise serializers.ValidationError("ths tweet is longer ")
        return value