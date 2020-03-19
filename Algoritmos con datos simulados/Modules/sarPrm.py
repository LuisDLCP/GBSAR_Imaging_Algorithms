# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:29:38 2018
-------------------------------------------------
      LIBRERÍA DE DEFINICION DE PARAMETROS
      PARA LOS ALGORITMOS DE IMAGENES SAR
-------------------------------------------------
@author: LUIS
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

show = False

# Funcion que define los parametros del sistema(Simulacion)
def get_parameters():
    # Definición de parámetros
    c = 0.3 #0.299792458 # Velocidad de la luz (x 1e9 m/s)
    fc = 15 # Frecuencia Central(GHz)
    BW = 0.6 # Ancho de banda(GHz)
    Ls = 1.2 # 4 # 0.6 # Longitud del riel (m)
    Ro = 0 # Constante debido al delay de los cables(m)
    theta = 75 #90 # Angulo azimuth de vision de la imagen final(grados sexagesimales E [0-90])

    # Definicion de las dimensiones de la imagen ("dh" y "dw" usados solo en BP)
    w = 10 # 800 # 5 # Ancho de la imagen(m)
    h = 10 #10 # 800 # 5 # Altura de la imagen(m)
    dw = 0.25 #0.25 # 0.2 #0.1 # Paso en el eje del ancho(m)
    dh = 0.25 #0.25 # 0.2 #0.1 # Paso en el eje de la altura(m)

    # Hallando el Np a partir de los pasos
    dp=c/(4.1*fc*np.sin(theta*np.pi/180)) # paso del riel para un angulo de vision de 180°
    Np=int(Ls/dp)+1 # Numero de pasos del riel

    if Np%2!=0:
        Np+=1   # Para que el numero de pasos sea par

    #dp = 0.6
    #Np=int(Ls/dp)+1
    #Nf=2**10

    # Hallando el Nf en funcion a la distancia máxima deseada
    r_r=c/(2*BW) # resolucion en rango
    Nf=int(h/r_r) +1 #Numero de frecuencias

    prm={
        'c':c,
        'fc':fc,
        'BW':BW,
        'Ls':Ls,
        'Ro':Ro,
        'theta':theta,
        'Np':Np,
        'Nf':Nf,
        'w':w,
        'h':h,
        'dw':dw,
        'dh':dh
    }
    return prm

# Funcion que define los parametros del sistema(Datos reales)
def get_parameters2(dset):
    # Definición y lectura de parámetros
    c = 0.299792458 # Velocidad de la luz (x 1e9 m/s)
    fi = dset.attrs['fi']*1e-9 # Frecuencia inferior(GHz)
    fs = dset.attrs['ff']*1e-9 # Frecuencia superior(GHz)
    Nf = dset.attrs['nfre'] # Numero de frecuencias
    xi = float(dset.attrs['xi']) # Posicion inicial(m)
    xf = float(dset.attrs['xf']) # Posicion final(m)
    Np = dset.attrs['npos'] # Numero de posiciones
    dx = float(dset.attrs['dx']) # Paso del riel(m)
    b_angle = dset.attrs['beam_angle']/2 # Angulo del haz(°)

    fc = (fi+fs)/2 # Frecuencia Central(GHz)
    BW = fs-fi # Ancho de banda(GHz)
    Ls = xf-xi # Longitud del riel (m)
    Ro = 0 # Constante debido al delay de los cables(m)

    # Definicion de las dimensiones de la imagen ("dh" y "dw" usados solo en BP)
    w = 700 # Ancho de la imagen(m)
    h = 850 # Altura de la imagen(m)
    hi = 100 # Posicion inicial en el eje del rango
    dw = 0.5 # Paso en el eje del ancho(m)
    dh = 0.5 # Paso en el eje de la altura(m)

    prm={
        'c':c,
        'fc':fc,
        'BW':BW,
        'Ls':Ls,
        'Ro':Ro,
        'theta':b_angle,
        'Np':Np,
        'Nf':Nf,
        'dx':dx,
        'w':w,
        'h':h,
        'hi':hi,
        'dw':dw,
        'dh':dh
    }
    return prm

#--------FUNCIONES PARA DEFINIR POSICION Y MAGNITUD DE LOS TARGETS SIMULADOS-------------
# Un solo target
def get_scalar_data():
    at=np.array([1]) # Reflectividad
    Rt=np.array([(0,5)]) # Coordenadas del target (x,y)m
    return at,Rt

# Varios targets
def get_scalar_data2():
    at=np.array([1,1,1]) # Reflectividad
    Rt=np.array([(0,2),(4,8),(-4,8)]) # Coordenadas del target (x,y)m
    return at,Rt

def get_matrix_data1():
    # Se define una imagen con la letra T

    ro_x, ro_y = -2.5, 2 # Posicion inicial (x,y)m
    d_x, d_y = 0.25,0.25#0.125,0.125 #0.25, 0.25 # Paso en los ejes (x,y)m
    N_r, N_c = 21,21#41,41 #21, 21 # Numero de filas y columnas de la matriz de targets
    Data = np.array([[0]*7+[1]*7+[0]*7]*14+[[1]*N_c]*7)#np.array([[0]*13+[1]*15+[0]*13]*28+[[1]*N_c]*13) #np.array([[0]*4+[1]*3+[0]*4]*7+[[1]*N_c]*4)
    I_t = Data.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) #  Vector de coordenadas (x,y)

    # Grafica de la imagen a simular
    if show:
        fig, ax = plt.subplots()
        ax.imshow(Data,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
        ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt

def get_matrix_data2():
    # Se define la imagen del logo de Superman

    image_file = Image.open("superman.jpg") # open colour image
    image_file = image_file.convert('L') # convert image to grayscale
    imgData1 = np.asarray(image_file)
    imgData1 = (imgData1<(imgData1.max()-imgData1.min())/2)*1 # Imagen final en blanco y negro
    imgData1 = np.flipud(imgData1)
    # Define las intensidades y coordenadas
    ro_x, ro_y = -4,3 #-6,5 # Posicion inicial (x,y)m
    N_r, N_c= len(imgData1), len(imgData1[0]) # Numero de filas y columnas
    d_x, d_y= 8/N_c, 8/N_r #0.125, 0.125 #8/N_c, 12/N_r # Paso en los ejes (x,y)m

    I_t=imgData1.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) # Coordenada (x,y) de los targets

    # Grafica de la imagen a simular
    if show:
        fig, ax = plt.subplots()
        ax.imshow(imgData1,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
        ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt

def get_matrix_data3():
    # Se define una imagen con la letra ROJ

    ro_x, ro_y = -5, 3 # Posicion inicial (x,y)m
    d_x, d_y = 0.25,0.25#0.125,0.125 #0.25, 0.25 # Paso en los ejes (x,y)m
    N_r, N_c = 16,41

    # Formando las letras x separado
    # Letra R
    #letraR = np.array([[1]*3+[0]*6+[1]*2]*2+[[1]*3+[0]*4+[1]*2+[0]*2]*2+[[1]*3+[0]*2+[1]*2+[0]*4]*2+[[1]*5+[0]*6]*2+[[1]*11]*8)
    letraR = np.array([[1]*3+[0]*6+[1]*2]*2+[[1]*3+[0]*4+[1]*2+[0]*2]*2+[[1]*3+[0]*2+[1]*2+[0]*4]*2+[[1]*5+[0]*6]*2+[[1]*11]*3+[[1]*3+[0]*5+[1]*3]*2+[[1]*11]*3)
    # Letra 0
    #letraO = np.ones((16,11))
    letraO = np.array([[1]*11]*3+[[1]*3+[0]*5+[1]*3]*10+[[1]*11]*3)
    # Letra J
    letraJ = np.array([[1]*9+[0]*2]*3+[[1]*3+[0]*3+[1]*3+[0]*2]*2+[[0]*6+[1]*3+[0]*2]*8+[[1]*11]*3)
    # Joining all the letters
    Data = np.zeros((16,41))
    Data[:,2:13] = letraR
    Data[:,15:26] = letraO
    Data[:,28:39] = letraJ

    I_t = Data.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) #  Vector de coordenadas (x,y)

    # Grafica de la imagen a simular
    if show:
        fig, ax = plt.subplots()
        im=ax.imshow(Data,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
        ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt

def get_matrix_data4():
    # Se define la imagen de un cerro

    image_file = Image.open("mountain3.jpg") # open colour image
    image_file = image_file.convert('L') # convert image to grayscale
    imgData1 = np.asarray(image_file)
    imgData1 = (imgData1<(imgData1.max()-imgData1.min())/2)*1 # Imagen final en blanco y negro
    imgData1 = np.flipud(imgData1)
    # Define las intensidades y coordenadas
    ro_x, ro_y = -4.5,1 #-6,5 # Posicion inicial (x,y)m
    N_r, N_c= len(imgData1), len(imgData1[0]) # Numero de filas y columnas
    d_x, d_y= 8/N_c, 8/N_r #0.125, 0.125 #8/N_c, 12/N_r # Paso en los ejes (x,y)m

    I_t=imgData1.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) # Coordenada (x,y) de los targets

    # Grafica de la imagen a simular
    if show:
        fig, ax = plt.subplots()
        ax.imshow(imgData1,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
        ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt

def get_matrix_data5():
    # Se define una imagen con la letra UNI

    ro_x, ro_y = -5, 3 # Posicion inicial (x,y)m
    d_x, d_y = 0.25,0.25#0.125,0.125 #0.25, 0.25 # Paso en los ejes (x,y)m
    N_r, N_c = 16,41

    # Formando las letras x separado
    # Letra U
    letraU = np.array([[1]*11]*3+[[1]*3+[0]*5+[1]*3]*13)
    # Letra N
    letraN = np.array([[1]*16]*3+[[0]*10+[1]*4+[0]*2]*1+[[0]*8+[1]*4+[0]*4]*1+[[0]*6+[1]*4+[0]*6]*1+[[0]*4+[1]*4+[0]*8]*1+[[0]*2+[1]*4+[0]*10]*1+[[1]*16]*3).T
    # Letra I
    letraI = np.array([[1]*11]*3+[[0]*4+[1]*3+[0]*4]*10+[[1]*11]*3)

    # Joining all the letters
    Data = np.zeros((16,41))
    Data[:,2:13] = letraU
    Data[:,15:26] = letraN
    Data[:,28:39] = letraI

    I_t = Data.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) #  Vector de coordenadas (x,y)

    # Grafica de la imagen a simular
    if show:
        fig, ax = plt.subplots()
        im=ax.imshow(Data,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
        ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt
