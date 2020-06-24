from controller import Robot
import cv2 as cv
import numpy as np

## tiempo sampleo de la simulacion
TIME_STEP = 10
robot = Robot()
## parte de los sensores de distancia
ds = []
dsNames = ['ps0', 'ps7']
for i in range(2):
    ds.append(robot.getDistanceSensor(dsNames[i]))
    ds[i].enable(TIME_STEP)
##acceso a los nodos de vision del robot
camera_id=[]    
camara=['camera']
for i in range(1):
    camera_id.append(robot.getCamera(camara[i]))
    camera_id[i].enable(TIME_STEP)
## parte para actuadores del robots
wheels = []
wheelsNames = ['left wheel motor', 'right wheel motor']
for i in range(2):
    wheels.append(robot.getMotor(wheelsNames[i]))
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0.0)
avoidObstacleCounter = 0
## datos para la deteccion de bolas azules
## componenetes en rojo a filtrar
redbajo1=np.array([0,100,20],np.uint8)
redAlto1=np.array([8,255,255],np.uint8)

redbajo2=np.array([175,100,20],np.uint8)
redAlto2=np.array([179,255,255],np.uint8)
#3 componentes en azul a filtrar
azulbajo=np.array([115,100,20],np.uint8)
azulalto=np.array([140,255,255],np.uint8)
## matri de trnaformacion para el sistema de cordenadas
T=np.matrix([[0,1,0,-400],[-1,0,0,150],[0,0,1,0],[0,0,0,1]])
## matriz de transformacion para velocidades generales
r=0.0205
L=0.0710
T_vel=np.matrix([[1/r,L/(2*r)],[1/r,-L/(2*r)]])
u=0
w=0
generales=np.matrix([[u],[w]])
error_1=0;
## bucle de simulacion de robot
while robot.step(TIME_STEP) != -1:
 
    ## obtencion de la imagen del robot
    image = camera_id[0].getImageArray()
    fer=np.uint8(image)
    fer=cv.rotate(fer,cv.ROTATE_90_CLOCKWISE)
    b,g,r=cv.split(fer)
    ## imagen final para ahcer el control
    procesamiento=cv.merge([r,g,b])
    procesamiento=cv.flip(procesamiento,1)
    ## filtro color rojo por hsv
    frameHSV=cv.cvtColor(procesamiento,cv.COLOR_BGR2HSV)
    maskRed1=cv.inRange(frameHSV,redbajo1,redAlto1)
    maskRed2=cv.inRange(frameHSV,redbajo2,redAlto2)
    maskRed=cv.add(maskRed1,maskRed2)
    maskRedvis=cv.bitwise_and(procesamiento,procesamiento,mask=maskRed)
    ## filtro para el color azul hsv
    maskBlue=cv.inRange(frameHSV,azulbajo,azulalto)
    maskBluevis=cv.bitwise_and(procesamiento,procesamiento,mask=maskBlue)
    ## contornos del sistema
    contours, hierarchy=cv.findContours(maskBlue,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area=cv.contourArea(c)
        if area>1000:
            M=cv.moments(c)
            if (M["m00"]==0): M["m00"]=1
            x=int(M["m10"]/M["m00"])
            y=int(M["m01"]/M["m00"])
            cv.circle(procesamiento, (x,y), 7, (0,255,0), -1)
            font = cv.FONT_HERSHEY_SIMPLEX
            contorno_limpio=cv.convexHull(c)
            x_real=y
            y_real=x
            #print(x_real,y_real)
            cv.drawContours(procesamiento, [contorno_limpio], 0, (0,0,0), 3)
            entrada=np.matrix([[x_real],[y_real],[0],[1]])
            sistema_real=T@entrada
            cv.putText(procesamiento, '{},{}'.format(sistema_real[0,0],sistema_real[1,0]),(x+10,y), font, 0.75,(0,255,0),1,cv.LINE_AA)
            #print(entrada,sistema_real)
            x_d=0;
            error=x_d-sistema_real[0,0]
            w_control=0.2*np.tanh(error)
            if abs(error)<5:
                w_control=0;
            velocities=T_vel@np.matrix([[0],[w_control]])
            print(velocities)
            leftSpeed = velocities[1,0]
            rightSpeed = velocities[0,0]
            ## aplicacion de veloicades a cada rueda
            wheels[0].setVelocity(leftSpeed)
            wheels[1].setVelocity(rightSpeed)
            
    #leftSpeed = 0.0
    #rightSpeed = 0.0
    ## aplicacion de veloicades a cada rueda
    #wheels[0].setVelocity(leftSpeed)
    #wheels[1].setVelocity(rightSpeed)
    #cv.drawContours(procesamiento, contours, -1, (0,0,0), 3)
    #uestra de resultados
    #filtro azul
    #cv.imshow("filtro azul",maskBluevis)
    #filtro rojo
    #cv.imshow("filtro rojo",maskRed)
    ## muestra de la imagen en pantalla
    cv.imshow('frame', procesamiento)
    #print(fer.shape,fer.dtype)
    if cv.waitKey(1) == ord('q'):
        break
    error_1=error
cap.release()
cv.destroyAllWindows() 