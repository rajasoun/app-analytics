#!/usr/bin/env sh

sudo date -u --set="$(curl -H 'Cache-Control: no-cache' -sD - http://google.com |grep '^Date:' |cut -d' ' -f3-6)"
