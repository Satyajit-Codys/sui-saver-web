import sys
import tweepy


def twitter_auth():
    try:
        consumer_key = "KllB6T1dVrLd3gFS0fRmziFiS"
        consumer_secret = "LDFfoLOKUyjGJ4s8I9BU7QhoclO45eSQRou7c5NCJnhAUWaUey"
        access_token = "829526406375809027-dGb1fcP8n0ctdeUCXSs6tY5art0fYjO"
        access_secret = "h288hE0iNPJITrvYCImZc2g8rur8AJv2RrilVEof4DGCr"
    except:
        sys.stderr.write("Twitter_* environment variable not set\n")
        sys.exit(1)
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth


def get_twitter_client():
    auth = twitter_auth()
    client = tweepy.API(auth, wait_on_rate_limit=True)
    return client


def get_tweets():
    tweets = []
    user = "satyabot1"
    n = 100
    client = get_twitter_client()
    for status in tweepy.Cursor(client.user_timeline, screen_name=user).items(n):
        # print(status.text)
        tweets.append(status.text)
    return(tweets)
