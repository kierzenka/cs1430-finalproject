# gsv_pieces = ['Pnt_start0_end1000','Pnt_start1000_end2000','Pnt_start2000_end3000','Pnt_start3000_end4000']
gsv_pieces = ['Pnt_start0_end1000','Pnt_start1000_end2000']

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


with open("provPredictionsNew.csv", 'w') as writ:
  for chunk in gsv_pieces:
    with open("/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/provLabelsNew/"+chunk+"_labels.txt", 'r') as f:
      lines = f.readlines()
      for l in lines:
        id = l.split(",")[0]
        gvi = l.split(",")[1]
        # lat = id_to_lat[id]
        # long = id_to_long[id]
        writ.write(lat+","+long+","+gvi)
