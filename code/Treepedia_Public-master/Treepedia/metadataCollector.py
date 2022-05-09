
# This function is used to collect the metadata of the GSV panoramas based on the sample point shapefile

# Copyright(C) Xiaojiang Li, Ian Seiferling, Marwa Abdulhai, Senseable City Lab, MIT 

# from importlib.metadata import metadata


def GSVpanoMetadataCollector(samplesFeatureClass,num,ouputTextFolder):
    '''
    This function is used to call the Google API url to collect the metadata of
    Google Street View Panoramas. The input of the function is the shpfile of the create sample site, the output
    is the generate panoinfo matrics stored in the text file
    
    Parameters: 
        samplesFeatureClass: the shapefile of the create sample sites
        num: the number of sites proced every time
        ouputTextFolder: the output folder for the panoinfo
        
    '''

    import cStringIO
    from osgeo import ogr, osr
    import time
    import os,os.path
    import google_streetview.api
    import json
    
    if not os.path.exists(ouputTextFolder):
        os.makedirs(ouputTextFolder)
    
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    dataset = driver.Open(samplesFeatureClass)
    layer = dataset.GetLayer()

    # loop all the features in the featureclass
    feature = layer.GetNextFeature()
    featureNum = layer.GetFeatureCount()
    batch = featureNum/num
    
    for b in range(batch):
        # for each batch process num GSV site
        start = b*num
        end = (b+1)*num
        if end > featureNum:
            end = featureNum
        
        ouputTextFile = 'Pnt_start%s_end%s.txt'%(start,end)
        ouputGSVinfoFile = os.path.join(ouputTextFolder,ouputTextFile)
        
        # skip over those existing txt files
        if os.path.exists(ouputGSVinfoFile):
            continue
        
        time.sleep(1)
        
        with open(ouputGSVinfoFile, 'w') as panoInfoText:
            # process num feature each time
            for i in range(start, end):
                feature = layer.GetFeature(i)      
                geom = feature.GetGeometryRef()
                lon = geom.GetX()
                lat = geom.GetY()
                
                # Define parameters for street view api
                params = [{
                'size': '600x300', # max 640x640 pixels
                'location': '%s,%s'%(lat, lon),
                'key': 'AIzaSyDHoCT01Ixth_JKqK7k_lkkr00mpDoNDF0'
                }]
                
                #get image
                results = google_streetview.api.results(params)
                
                #save pictures for verification
                results.download_links('/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/links')
                
                #get metadata as json, convert to dict
                results.save_metadata('/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/metadataNew.json')
                with open('/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/metadataNew.json') as f:
                    data1 = json.load(f)
                data = data1[0]

                # in case there is not panorama in the site, therefore, continue
                if data['pano_id']==None:
                    print('no panorama')
                    continue
                else:
                    # panoInfo = data['panorama']['data_properties']
                    print('getting metadata')         
                    # get the meta data of the panorama
                    panoDate = data['date']
                    panoId = data['pano_id']
                    panoLat = data['location']['lat']
                    panoLon = data["location"]['lng']
                    
                    print('The coordinate (%s,%s), panoId is: %s, panoDate is: %s'%(panoLon,panoLat,panoId, panoDate))
                    lineTxt = 'panoID: %s panoDate: %s longitude: %s latitude: %s\n'%(panoId, panoDate, panoLon, panoLat)
                    panoInfoText.write(lineTxt)
                    
        panoInfoText.close()


# ------------Main Function -------------------    
if __name__ == "__main__":
    import os, os.path
    
    inputShp = '/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/provPointsNew/provPointsNew.shp'
    outputTxt = '/Users/alexkamper/Desktop/cs1430-finalproject/data/provData/metaDataNew'
    
    GSVpanoMetadataCollector(inputShp,1000,outputTxt)

