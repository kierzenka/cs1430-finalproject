import google_streetview.api
import google_streetview.helpers
import shutil
import os

output_path = "/Users/alexkamper/Desktop/cs1430-finalproject/data/provData"
section_path = "provImages"
# with open("./metaDataNew/"+section_path+".txt", 'r') as f:
with open("/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/metaDataNew/Pnt_start5000_end6000.txt", 'r') as f:
    line_list = f.readlines()
    print(len(line_list))
    pano_list = ""
    for l in line_list:
        panoID = l.split(":")[1].split()[0].strip()
        pano_list = pano_list + ";" + panoID
    
        params = [{
        'size':'244x244',
        'pano':panoID,
        'key':'AIzaSyCKNesxqMcBamAoPcNRR2YoxBgeQ60SOT0'}]
        
        results = google_streetview.api.results(params)
        print('downloaded')
        results.download_links("/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/temp")
        os.rename("/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/temp/gsv_0.jpg", "/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/provImages/"+panoID+".jpg")
        # shutil.move(output_path+"/temp/gsv_0.jpg", output_path+"/"+section_path+"/"+panoID+".jpg") 
    #print(l)
    #print(l.split(":"))
    #print(l.split(":")[1].split()[0].strip())
