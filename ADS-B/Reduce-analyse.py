#! /usr/bin/env python3

import sys

path = []
lastflight = None
count = 0

for line in sys.stdin:
	line = line.strip()

	icao, lat, lon = line.split()
	
	if lastflight is None:
		lastflight = icao
	if icao == lastflight:
		if (lat, lon) not in path:
			path.append((lat, lon))
			count++
	else:
		print("%s\t%s" % (icao, count))
		path=[]
		lastflight = icao
		count = 0
	if lastflight is not None:
		print("%s\t%s" % (icao, count))

	