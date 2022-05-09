gsv_pieces = ['Pnt_start0_end1000','Pnt_start1000_end2000','Pnt_start2000_end3000','Pnt_start3000_end4000']

id_to_coor = {}
for chunk in gsv_pieces:
    with open("../data/provData/metadataNew/"+chunk+".txt", 'r') as f:
      lines = f.readlines()
      for l in lines:
        id = l.split(":")[1].split()[0].strip()
        long = l.split(":")[3].split()[0].strip()
        lat = l.split(":")[4].split()[0].strip()
        id_to_coor[id] = (lat,long)

with open("provPredictions.csv", 'w') as writ:
  for chunk in gsv_pieces:
    with open("../data/provData/"+chunk+"_labels.txt", 'r') as f:
      lines = f.readlines()
      for l in lines:
        id = l.split(",")[0]
        gvi = l.split(",")[1]
        lat,long = id_to_coor[id]
        writ.write(lat+","+long+","+gvi)
