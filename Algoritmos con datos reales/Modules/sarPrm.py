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

# Funcion que define los parametros del sistema(Simulacion)
def get_parameters():
    # Definición de parámetros
    c = 0.3 # 0.299792458 # Velocidad de la luz (x 1e9 m/s)
    fc = 15 # Frecuencia Central(GHz)
    BW = 0.6   # Ancho de banda(GHz)
    Ls = 0.5 # Longitud del riel (m)
    Ro = 0 # Constante debido al delay de los cables(m)
    theta = 90 # Angulo azimuth de vision de la imagen finaL(grados sexagesimales E [0-90])

    # Definicion de las dimensiones de la imagen ("dh" y "dw" usados solo en BP)
    w = 8 # 800 # 5 # Ancho de la imagen(m)
    h = 8 # 800 # 5 # Altura de la imagen(m)
    dw = 0.025 # 0.2 #0.1 # Paso en el eje del ancho(m)
    dh = 0.025 # 0.2 #0.1 # Paso en el eje de la altura(m)

    # Hallando el Np a partir de los pasos
    dp=c/(8*fc) # paso del riel para un angulo de vision de 180°
    Np=int(Ls/dp)+1 # Numero de pasos del riel

    if Np%2!=0:
        Np+=1   # Para que el numero de pasos sea par

    # Hallando el Nf en funcion a la distancia máxima deseada
    r_r=c/(2*BW) # resolucion en rango
    Nf=int(h/r_r)+1 # +1 Numero de frecuencias

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
    time = dset.attrs['datetime'] # Fecha de toma de datos 

    fc = (fi+fs)/2 # Frecuencia Central(GHz)
    BW = fs-fi # Ancho de banda(GHz)
    Ls = xf-xi # Longitud del riel (m)
    Ro = 0 # Constante debido al delay de los cables(m)
    R_max=Nf*c/(2*BW) # Distancia maxima teorica 

    # Definicion de las dimensiones de la imagen ("dh" y "dw" usados solo en BP)
    w = 400 # Ancho de la imagen(m)
    h = 650 # Altura de la imagen(m)
    hi = 300 # Posicion inicial en el eje del rango
    dw = 1 # Paso en el eje del ancho(m)
    dh = 1 # Paso en el eje de la altura(m)

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
        'dh':dh,
        'Rmax':R_max,
        'date':time
    }
    return prm

#--------FUNCIONES PARA DEFINIR POSICION Y MAGNITUD DE LOS TARGETS-------------
# Un solo target
def get_scalar_data():
    at=np.array([1]) # Reflectividad
    Rt=np.array([(2,4)]) # Coordenadas del target (x,y)m
    return at,Rt

# Varios targets
def get_scalar_data2():
    at=np.array([1,1]) # Reflectividad
    Rt=np.array([(0,1),(0,1.25)]) # Coordenadas del target (x,y)m
    return at,Rt

# Matriz de targets tipo 1: el usuario crea su matriz de intensidades
def get_matrix_data1():
    # Define la matriz de intensidades I_t:(LETRA T)
    #       [[1 1 1 1 1 1 1 1 1 1 1]
    #        [1 1 1 1 1 1 1 1 1 1 1]
    #        [1 1 1 1 1 1 1 1 1 1 1]
    #        [0 0 0 1 1 1 1 1 0 0 0]
    #        [0 0 0 1 1 1 1 1 0 0 0]
    #        [0 0 0 1 1 1 1 1 0 0 0]
    #        [0 0 0 1 1 1 1 1 0 0 0]
    #        [0 0 0 1 1 1 1 1 0 0 0]
    #        [0 0 0 1 1 1 1 1 0 0 0]
    #        [0 0 0 1 1 1 1 1 0 0 0]
    #        [0 0 0 1 1 1 1 1 0 0 0]]

    ro_x, ro_y = -2, 2 # Posicion inicial (x,y)m
    d_x, d_y = 0.5, 0.5 # Paso en los ejes (x,y)m
    N_r, N_c = 11, 11 # Numero de filas y columnas de la matriz de targets
    Data = np.array([[0]*4+[1]*3+[0]*4]*7+[[1]*N_c]*4)
    I_t = Data.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) #  Vector de coordenadas (x,y)

    # Grafica de la imagen a simular
    fig, ax = plt.subplots()
    im=ax.imshow(Data,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
    ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt

# Matriz de targets tipo 2: la matriz de intensidades se crea a partir de una imagen cargada
def get_matrix_data2():
    # Carga y convierte la imagen a escala de grises
    image_file = Image.open("superman.jpg") # open colour image
    image_file = image_file.convert('L') # convert image to grayscale
    imgData1 = np.asarray(image_file)
    imgData1 = (imgData1<(imgData1.max()-imgData1.min())/2)*1 # Imagen final en blanco y negro
    imgData1 = np.flipud(imgData1)
    # Define las intensidades y coordenadas
    ro_x, ro_y = -4, 1 # Posicion inicial (x,y)m
    N_r, N_c= len(imgData1), len(imgData1[0]) # Numero de filas y columnas
    d_x, d_y= 8/N_c, 12/N_r # Paso en los ejes (x,y)m

    I_t=imgData1.reshape(1,N_r*N_c)[0] # Vector de intensidades

    rt_x=ro_x+d_x*np.arange(N_c) # Coordenada x de los targets(empezando en ro_x)
    rt_y=ro_y+d_y*np.arange(N_r) # Coordenada y de los targets(empezando en ro_y)
    rt=np.array([(x,y) for y in rt_y for x in rt_x]) # Coordenada (x,y) de los targets

    # Grafica de la imagen a simular
    fig, ax = plt.subplots()
    im=ax.imshow(imgData1,aspect='auto',origin='lower',extent=[rt_x.min(), rt_x.max(), rt_y.min(),rt_y.max()])
    ax.set(xlabel='Eje x(m)',ylabel='Eje y(m)', title='Imagen para la simulación')

    return I_t, rt
