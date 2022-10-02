
import copy
import argparse
import glob
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plot
import sympy




def compute_hcf(x, y):
    while(y):
        x, y = y, x % y
    return x

def adaptive_his(image,window,climit,channel=None):
    if channel == None:
        img = copy.deepcopy(image)
    else:
        img = image[:, :, channel]
        
    i,j = img.shape 
    f = window
    w, h = i//f, j//f

    for a in range(f):
        for b in range(f):
            temp = img[a*w:a*w+w, b*h:b*h+h]
            
            img[w*a:w*a+w, h*b:h*b+h] = histogram(temp,climit=climit)
            
    return img

         
def window(img,flag):
    img[:,:2] = 0
    # img = np.flipud(img)
    cdst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    way = []
    
    dict = {}
    i,j =  img.shape   
    s = i//20
    
    for k  in range(j):
        his = cv2.calcHist([img[:,k]],[0],None,[256],[0,256])
        dict[k] = his[-1].tolist()[0]

    solid = max(dict, key=dict.get)
    xpts = []
    ypts = []
    for k in range(20):
        win = img[i-(k+1)*s:i-(k)*s,solid-4:solid+4]
        p =  np.where(win > 0)
        
        if len(p[1]) != 0:
            p = [l+solid-4 for l in p[1]]
            cdst = cv2.rectangle(cdst,((solid-10),i-k*s),((solid+10),round(i-(k+1)*s)),(255,255,0),1)
            solid = round(np.mean(p))
            xpts.append(round(i-(k+0.5)*s))
            ypts.append(solid)

    

            
            
    R = 0
    xp = np.linspace(0,i-1,i)
    if len(xpts)>12:
        
        if flag != 1:
            coef = np.polyfit(xpts,ypts,2)

            R = curvature(coef,j)
    
            
            yp = [coef[0]*n**2 + coef[1]*n + coef[2] for n in xp]
            points = [(yp[v],xp[v]) for v in range(len(xp))]

        else:
            coef = np.polyfit(xpts,ypts,1)

            yp = [coef[0]*n + coef[1] for n in xp]
            points = [(yp[v],xp[v]) for v in range(len(xp))]

            
        
        return points,xpts,R,True,cdst
    else:
        return None,None,R,False,cdst
        

    



def  histogram(image,climit=None,channel=None):
    
    if channel == None:
        img = copy.deepcopy(image)
    else: 
        img = image[:,:,channel]
        
    intensity = {}

    pal = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i, j] in intensity) :
            
                temp = {img[i, j]: intensity.get(img[i, j])+1}
                intensity.update(temp)
            
            else:
                
                intensity[img[i, j]] = 1
    if climit!=None:      
        for key in intensity:
            if intensity.get(key)>climit:
                temp = {key:climit}
                pal += intensity.get(key)-climit
                intensity.update(temp)
            else:
                continue
        
        
        while pal > 0:
            
            for key in range(256):
                if key in intensity:
                    temp = {key: intensity.get(key)+1}
                    intensity.update(temp)
                    pal-=1
                
                else:
                    intensity[key] = 1
                    pal-=1
                if pal == 0:
                    break 
    
    # plot.plot(*zip(*sorted(intensity.items())))
    # plt.savefig(path + "histogram.png")

    intensity = {key: intensity[key] for key in sorted(intensity.keys())}
    N = img.shape[0]*img.shape[1]
    h = 0
    for key in intensity:
        h += intensity.get(key)
        temp = {key: h/N}
        intensity.update(temp)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = intensity.get(img[i,j]) * 255
    return img
def dis(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def warp_points(img_size):

    height = img_size[0]
    width = img_size[1]

    s1 = [width // 2 - 100, height * 0.625]
    s2 = [width // 2 + 100, height * 0.625]
    s3 = [0, height-50]
    s4 = [width , height-50]
    src = np.float32([s1, s2, s4, s3])

    d1 = [0, 0]
    d2 = [dis(s1,s2), 0]
    d3 = [0, dis(s2,s4)]
    d4 = [dis(s1, s2), dis(s2, s4)]
    dst = np.float32([d1, d2, d4, d3])

    return src, dst

def curvature(coef,j):
    a,b,c = coef[0],coef[1],coef[2]
    x = sympy.symbols('x')
    

    eq = a*x**2+ b*x + c
    deq = sympy.diff(eq)
    ddeq = sympy.diff(deq)
    R = math.sqrt((1+ddeq**2)**3)/abs(deq)
    R = R.subs(x,j)
    
    
    return R



def sobel(img):

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

#############################
###### Problem 1 ############
#############################

#######################################
path = "/home/naveen/ENPM673/namngla_project2/"
#######################################


# Parser = argparse.ArgumentParser()
# Parser.add_argument('--BasePath', default='/home/naveen/ENPM673/project2/adaptive_hist_data',
#                     help='Give your path')

# Args = Parser.parse_args()
# BasePath = Args.BasePath

# images = [cv2.imread(file) for file in sorted(glob.glob(str(BasePath)+'/*.png'))]
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# i, j, k = images[0].shape
# out1 = cv2.VideoWriter(path + 'normal.mp4', fourcc, 5.0, (j, i))

# out = cv2.VideoWriter(path + 'adaptive.mp4', fourcc, 5.0, (j, i))

# kernel = (1/8)*np.array([[0,1,0],[1,4,1],[0,1,0]])

# for im in images:
#     img = copy.deepcopy(im)
#     his = copy.deepcopy(im)
#     for c in range(3):
#         his[:,:,c] = histogram(his,channel=c)
#         img[:,:,c] = adaptive_his(img,8,60,channel=c)
    
    
#     im_v = cv2.vconcat([img,his,im])
    
    
#     cv2.imwrite(path+"Acorrected.png", img)
#     out1.write(his)
    
#     out.write(img)
    
# out1.release()
# out.release()

##########   Problem 2,3 ################

flag =  0  ########## Change flag for desired output

if flag ==1:
    output = "lane1.mp4"
   
    cap = cv2.VideoCapture(path+'whiteline.mp4')
    
else:
    output = "lane2.mp4"
    cap = cv2.VideoCapture(path+'challenge.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)
lanevid= cv2.VideoWriter(path + output, fourcc, fps, (width, height))


######################################
ret = True
frames = []
his = [] 
while(ret):
  ret, frame = cap.read()
  if ret == True:
    frames.append(frame)

n = 0
for frame in frames:
    n+=1
    test = copy.deepcopy(frame)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    src,dest=warp_points(img.shape)
    
    src = src.reshape((-1, 1, 2))
    cv2.polylines(frame, [np.int32(src)], True, (0, 0, 255), 3)
    cv2.imwrite(path + "frame.png",frame)
    H, _ = cv2.findHomography(src, dest)
    
    out = cv2.warpPerspective(
        img, H, (int(dest[2][0]), int(dest[2][1])), flags=cv2.INTER_LINEAR)
    cv2.imwrite(path + "out.png",out)

   
    dst = cv2.Canny(out,150,255)
    cv2.imwrite(path + "canny.png",dst)

    kernel = np.ones((5,5),np.uint8)
    dst =  cv2.dilate(dst, kernel, iterations=1)
    cv2.imwrite(path + "dialate.png",dst)

    a,b = dst.shape

    rpoints,r,RR,detectr,right= window(dst[:,:b//2],flag)
    lpoints,c,RL,detectl,left= window(dst[:,b//2:],flag) 
    cv2.imwrite(path+"window2.png",cv2.hconcat([right,left]))

    print('Left R =' + str(RR)+"  Right R = "+str(RL))
    if detectr and detectl:         
        lpoints = [(p[0]+b//2,p[1]) for p in lpoints]
        his = []
        his.append(rpoints)
        his.append(r)
    
        his.append(lpoints)
        his.append(c)
        his.append(RR)

    else:
        rpoints = his[0]
        r = his[1]
        lpoints = his[2]
        c = his[3]
        RR = his[4]


    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    
    
    
    lane = np.zeros_like(cdst)
    
    ben = np.argmax([len(r),len(c)])
    
    if ben ==0:
        c1 = (0,0,255)
        c2 = (0, 255,0)
    else:
        c1 = (0,255, 0)
        c2 = (0, 0, 255)
    for p in rpoints:
        cv2.circle(lane,(int(p[0]),int(p[1])),2,c2,-1)
    for p in lpoints:
        cv2.circle(lane,(int(p[0]), int(p[1])), 2, c1, -1)

    
    
    warpes = cv2.warpPerspective(
        lane, np.linalg.inv(H), (test.shape[1],frame.shape[0]), flags=cv2.INTER_LINEAR)
    test = cv2.addWeighted(test.astype(np.uint8),0.5,warpes.astype(np.uint8),0.5,0)

    if RR>0:
        text = 'Turn Right'
    elif RR<0:
        text = 'Turn Left'
    else:
        text = ' Going Straight'
        
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    org = (5, 30)
    
    
    fontScale = 0.5
    
    
    color = (0, 0, 255)
    
    thickness = 2
    
  
    test = cv2.putText(test, text, org, font, fontScale, 
                    color, thickness, cv2.LINE_AA, False)
    lanevid.write(test)

    cv2.imshow("lane2",test)
    cv2.waitKey(100)

lanevid.release()
    

    
        

 


