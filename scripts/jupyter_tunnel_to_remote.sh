#!/usr/bin/env sh
ssh_user_ip = $1
ssh -N -L 8888:localhost:8888 $ssh_user_ip
