
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=128,height=128):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

convertjpg('F:\\phone_pic\\IMG.jpg',"F:\\opencv_pic")
convertjpg('F:\\phone_pic\\Touxiang.jpg',"F:\\opencv_pic")
