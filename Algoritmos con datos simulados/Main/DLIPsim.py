#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:15:55 2019
-------------------------------------------------------------
      Algoritmo "Discrete Linear Inverse Problem"(DLIP)
                       (Simulated)
Descripcion:
 . El presente algoritmo asume las siguientes consideraciones:
    - Riel en el eje X.
 . Se reconstruyen tanto targets individuales como matrices de targets.
 . Pruebas con datos simulados.
-------------------------------------------------------------
@author: LDCP
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import sarPrm as sp # Library created to define sistem parameters
import drawFigures as dF # Library created to draw FFT functions
import SARdata as Sd # Library created to get theoric data(Phase Historic)
import timeit
from mpl_toolkits.axes_grid1 import make_axes_locatable

#-----------------------LECTURA DE PARÁMETROS-----------------------------
prm = sp.get_parameters()
c,fc,BW,Nf = prm['c'],prm['fc'],prm['BW'],prm['Nf']
Ls,Np,Ro,theta = prm['Ls'],prm['Np'],prm['Ro'],prm['theta']
Lx,Ly,dx,dy = prm['w'],prm['h'],prm['dw'],prm['dh'] # Dimensiones de la imagen

def get_SAR_data():
    # Cálculo de parámetros
    It, Rt = sp.get_scalar_data2() # Magnitud  y coordenadas del target respectivamente
    dp=Ls/(Np-1) # Paso del riel(m)
    fi=fc-BW/2 # Frecuencia inferior(GHz)
    fs=fc+BW/2 # Frecuencia superior(GHz)

    # Cálculo de las resoluciones
    rr_r=c/(2*BW) # Resolución en rango
    rr_a=(c/(2*Ls*fc))*Rt.T[1].max() # Resolución en azimuth

    #-----------------VERIFICACIÓN DE CONDICIONES------------------------
    # Rango máximo
    R_max=Nf*c/(2*BW)
    # Paso del riel máximo
    dx_max=c/(fc*4*np.sin(theta*np.pi/180)) # Theta en grados sexagesimales

    print("------------------------------------------------------")
    print("--------------INFORMACIÓN IMPORTANTE------------------")
    print("------------------------------------------------------")
    print("- Resolución en rango(m) : ", rr_r)
    print("- Resolución en azimuth(m): ", rr_a)
    print("------------------------------------------------------")
    print("- Rango máximo permitido(m): ", R_max)
    print("------------------------------------------------------")
    print("______¿Se cumplen las siguientes condiciones?_________")
    print("Rango máximo del target <= rango máximo?: ", Rt.T[1].max()<=R_max) # Ponerle un try-except
    print("Paso del riel <= paso máximo?: ", dp<=dx_max) # Evita el aliasing en el eje de azimuth
    print("------------------------------------------------------")

    #----------------OBTENCIÓN DEL HISTÓRICO DE FASE----------------------
    Ski = Sd.get_phaseH(prm,It,Rt)
    #np.save("RawData_prueba3",Ski)
    #Ski = np.load("RawData_prueba1.npy")

    #-----------------GRÁFICA DEL HISTÓRICO DE FASE-----------------------
    dF.plotImage(Ski, x_min=fi, x_max=fs, y_min=-Ls/2, y_max=Ls/2,xlabel_name='Frecuencia(GHz)', ylabel_name='Posicion del riel(m)', title_name='Histórico de fase',unit_bar='', origin_n='upper')

    return {'Ski':Ski, 'dp':dp, 'fi':fi, 'fs':fs, 'R_max':R_max}

def DLIP_Algorithm(data):
    """ Ejecuta el algoritmo basado en Problemas de Inversion """
    # Lectura de parametros
    Ski = data['Ski'].copy()
    fi = data['fi']
    fs = data['fs']

    #------------- PRIMERA ETAPA: Planteamiento del problema -------------
    # Get following matrix and vectors:
    #              [Es] = [A][X] ___________________
    #   Raw_data <--|      |--> SAR_System_Matrix  |--> Reflectivity
    #---------------------------------------------------------------------

    # Creation of vector S1
    S1 = np.rS1hape(Ski.T,(len(Ski)*len(Ski[0]),1)) # Convert Raw Data matrix into vector
    riel_p = np.linspace(-Ls/2, Ls/2, Np) # Vector de posiciones
    f = np.linspace(fi, fs, Nf) # Vector de frecuencias
    ks = 2*np.pi*f/c # Range domain wavenumber
    # d_ks = ks[1]-ks[0] # Step

    # Definition of coordinates, including last value(Lx/2 and Ly)
    x_c=np.arange(-Lx/2,Lx/2+dx,dx)
    y_c=np.arange(0,Ly+dy,dy)
    r_c=np.array([(i,j) for j in y_c for i in x_c])

    # Funcion para calcular la distancia entre una posición del riel y una coordenada de la imagen
    def distance_nk(r_n,x_k): # vector de coordenadas de la imagen, punto del riel "k"
        d=((r_n[0]-x_k)**2+(r_n[1])**2)**0.5
        return d

    # Creation of Matrix A
    A = np.zeros((len(Ski)*len(Ski[0]),len(r_c)), dtype=np.complex) # dim = [(Nf*Np) x N], N es el numero de subgrids
    for n in range(len(r_c)):
        m = 0
        for s in range(len(f)):
            for l in range(len(riel_p)):
                A[m,n] = np.exp(-2j*ks[s]*np.abs(distance_nk(r_c[n],riel_p[l])))#/(4*np.pi*distance_nk(r_c[n],riel_p[l]))**2
                m+=1
    del Ski

    # Creation of Vector X (Uncomment for verification)
    """M = np.zeros((len(y_c),len(x_c))) # Tamaño de la escena(numero de pixeles)
    M[0,0] = 1 # (x,y) = (-2,0)
    X = np.reshape(M,(len(M)*len(M[0]),1))

    # Verificacion
    Es2 = np.matrix(A)*np.matrix(X) # Cumple con un error maximo de 1e-4
    """

    #----------  SEGUNDA ETAPA: Planteamiento de la solucion------------
    #                         [X] = ([A]**-1)([Es])
    #-----------------------------------------------------------------

    # Finding pseudoinverse of A
    rg_m = LA.matrix_rank(A) # Range of Matrix A
    Ap = LA.pinv(A) # A.T*(A*A.T)**-1
    del A
    S2 = Ap.dot(S1) # Array
    S3 = np.reshape(S2,(len(y_c),len(x_c)))

    # Normalizando
    Im /= np.sqrt(np.sum(abs(S3)**2))

    return {'Im':Im}

def plot_image(data2):
    """ Grafica la magnitud de la imagen"""
    # a) DefImicion y lectura de parametros
    Im = data2['Im'].copy()

    # b) Grafica final(magnitud)
    cmap="plasma"
    vmin = -100 #dB
    vmax = -20
    dF.plotImage(Im,cmap=cmap,xlabel_name='Azimut(m)',ylabel_name='Rango(m)', title_name='Resultado Algoritmo basado en Problemas de Inversión',
                 x_min=-(Lx+dx)/2, x_max=(Lx+dx)/2, y_min=0-dy/2, y_max=Ly+dy/2,unit_bar='(dB)',log=True,vmin=vmin,
                 vmax=vmax)
    return 'Ok!'

def main():
    plt.close('all') # Cerrar todas las figuras previas

    start_time = timeit.default_timer()
    datos = get_SAR_data() # Obtiene el historico de fase
    print("Tiempo de simulación(IP): ",timeit.default_timer() - start_time," s")

    start_time = timeit.default_timer()
    d_p = IP_Algorithm(datos) # Implementa el algoritmo IP
    print("Tiempo del procesamiento(IP): ",timeit.default_timer() - start_time," s")

    plot_image(d_p) # Grafica de la magnitud

if __name__ == '__main__':
  main()
