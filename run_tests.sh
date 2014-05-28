#!/bin/bash

MAXCOUNT=20
count=1

rm -f out.txt
echo "-----------------"
while [ "$count" -le $MAXCOUNT ]      # Generate 10 ($MAXCOUNT) random integers.
do
  eval ./a.out "$count"
  let "count += 1"  # Increment count.
done
echo "-----------------"
