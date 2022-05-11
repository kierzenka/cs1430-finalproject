#This script is used to convert our "panoID,GVI" csv into a "lat,lon,GVI" csv for plotting the map

gsv_pieces = ['Pnt_start0_end1000','Pnt_start1000_end2000']

#Load all of the panoID->(lat,lon) mappings from the first csv
id_to_long = {}
id_to_lat = {}
for chunk in gsv_pieces:
    with open("/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/metaDataNew/"+chunk+".txt", 'r') as f:
      lines = f.readlines()
      for l in lines:
        id = l.split(":")[1].split()[0].strip()
        long = l.split(":")[3].split()[0].strip()
        lat = l.split(":")[4].split()[0].strip()
        id_to_long[id] = long
        id_to_lat[id] = lat

print('XXXXXXXXXX')
print(id_to_long[id])
print(id_to_lat[id])

#Read all the panoID->GVI results and write them to a new csv as (lat,lon)->GVI
with open("provPredictionsNew.csv", 'w') as writ:
  for chunk in gsv_pieces:
    with open("/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/provLabelsNew/"+chunk+"_labels.txt", 'r') as f:
      lines = f.readlines()
      for l in lines:
        id = l.split(",")[0]
        gvi = l.split(",")[1]
        writ.write(lat+","+long+","+gvi)
