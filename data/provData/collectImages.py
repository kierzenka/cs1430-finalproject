import google_streetview.api
import google_streetview.helpers
import shutil
output_path = "./prov_gsv_images"
section_path = "Pnt_start3000_end4000"
with open("./metadataOutput/"+section_path+".txt", 'r') as f:
    line_list = f.readlines()
    pano_list = ""
    for l in line_list:
        panoID = l.split(":")[1].split()[0].strip()
        pano_list = pano_list + ";" + panoID
    
        params = [{
        'size':'244x244',
        'pano':panoID,
        'key':'AIzaSyCKNesxqMcBamAoPcNRR2YoxBgeQ60SOT0'}]
        
        results = google_streetview.api.results(params)
        results.download_links(output_path+"/temp")
        shutil.move(output_path+"/temp/gsv_0.jpg", output_path+"/"+section_path+"/"+panoID+".jpg") 
    #print(l)
    #print(l.split(":"))
    #print(l.split(":")[1].split()[0].strip())
