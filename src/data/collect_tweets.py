# christina lu
# collect_tweets.py
# using getoldtweets3, collect historical tweets from users

import GetOldTweets3 as got
import csv
import codecs
import pandas as pd
import os, sys, re, getopt
import traceback
import time

username_file = "/shared/0/projects/terfspot/username_negative.csv"

def get_tweets(start, end):
    users = []

    with open(username_file, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            users.append(row[-1])

    for i in range(start - 1, end):
        user = str(users[i])
        get_user_tweets(user, i+1)

def get_user_tweets(user, i):
    try:
        tweetCriteria = got.manager.TweetCriteria().setUsername(user).setMaxTweets(100)
        filename = "/shared/0/projects/terfspot/negative_tweets/tweets_" + str(i) + "_" + user + ".csv"

        outfile = open(filename, "w+", encoding="utf8")
        outfile.write('date,username,to,replies,retweets,favorites,text,geo,mentions,hashtags,id,permalink\n')

        cnt = 0
        def receiveBuffer(tweets):
            nonlocal cnt

            for t in tweets:
                data = [t.date.strftime("%Y-%m-%d %H:%M:%S"),
                    t.username,
                    t.to or '',
                    t.replies,
                    t.retweets,
                    t.favorites,
                    '"'+t.text.replace('"','""')+'"',
                    t.geo,
                    t.mentions,
                    t.hashtags,
                    t.id,
                    t.permalink]
                data[:] = [i if isinstance(i, str) else str(i) for i in data]
                outfile.write(','.join(data) + '\n')

            outfile.flush()
            cnt += len(tweets)

            if sys.stdout.isatty():
                print("\rSaved %i"%cnt, end='', flush=True)
            else:
                print(cnt, end=' ', flush=True)

        print("Downloading tweets...")
        got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)

    except KeyboardInterrupt:
        print("\r\nInterrupted.\r\n")
        sys.exit()

    except Exception as err:
        print(traceback.format_exc())
        print(str(err))

    except SystemExit:
        t = 480
        while t > 0:
            sys.stdout.write('\rSleeping to avoid rate limit. : {}s'.format(t))
            t -= 1
            sys.stdout.flush()
            time.sleep(1)

    finally:
        if "outfile" in locals():
            outfile.close()
            print()
            print('Done. Output file generated "%s".' % filename)

get_tweets(2, 17091)
