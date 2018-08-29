import re
import sys

#Map

for line in sys.stdin:
  line = line.strip()
  val = [x.strip(' ') for x in [i.split('|') for i in line]
  (icao, lat, lon) = (val[2], val[3], val[4])
  if (lat != "NULL" & lon != "NULL"):
	(lat, lon) = float(lat, lon)
	print ("%s\t%s\t%s"%(icao, lat, lon))


	
