import pandas as pd
df = pd.read_csv('tubedata.csv', header=None)
df.head()

from collections import defaultdict
 
station_dict = defaultdict(list)
zone_dict = defaultdict(set)

# get data row by row
for index, row in df.iterrows():
  
  start_station = row[0]
  end_station = row[1]
  act_cost = int(row[3])

  zone1 = row[4]
  zone2 = row[5]

  # station dictionary of child station tuples (child_name, cost from parent to the child)
  # {"Mile End": [("Stepney Green", 2), ("Wembley", 1)]}
  station_list = station_dict[start_station]
  station_list.append((end_station, act_cost))

  # the following two lines add the other direction of the tube "step"
  station_list = station_dict[end_station]
  station_list.append((start_station, act_cost))
  
  # we add the main zone
  zone_dict[start_station].add(zone1)
  # we add the secondary zone
  if zone2 != "0":
    zone_dict[start_station].add(zone2)
    # if the secondary zone is not 0 it's the main zone for the ending station
    zone_dict[end_station].add(zone2)
  else:
    # otherwise the main zone for the ending station is the same as for the starting station
    zone_dict[end_station].add(zone1)