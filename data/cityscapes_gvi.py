import numpy as np
from PIL import Image

# print(lis)

# maskarray = np.asarray(Image.open(lis[1]))
# print(maskarray.shape)
# print(np.max(maskarray))
# print(np.sum(maskarray)*100/(255*maskarray.shape[0]*maskarray.shape[1]))
# print(np.sum(maskarray))

with open("cityscapes_labels.txt", 'w') as w:
  with open("paths.txt") as f:
    all_lines = f.readlines()
    for line in all_lines:
      lis = line.split()
      # print(lis)
      labelpath = lis[1].replace("\\","/")
      imgpath = labelpath[6:-24]
      imgpath = "leftImg8bit"+ imgpath + "leftImg8bit.png"
      img = Image.open("./leftImg8bit_trainvaltest/"+imgpath)
      img = img.resize((244,244))
      img.save("./leftImg8bit_trainvaltest/"+imgpath)
      # print(labelpath)
      # print(imgpath)


      maskarray = np.asarray(Image.open("./gtFine_trainvaltest/"+lis[1].replace("\\","/")))
      #divide by 255 since mask is 0-255
      gvi = np.sum(maskarray)/(255*maskarray.shape[0]*maskarray.shape[1])
      w.write(imgpath + "\t" + str(gvi) + "\n")
