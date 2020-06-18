#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:11:41 2018
-------------------------------------------------------------
      Algoritmo "Frequency Domain Back Projection"(FDBP)
                        (Simulated)
-------------------------------------------------------------
Descripcion:
 . El presente algoritmo asume las siguientes consideraciones:
    - Riel en el eje X
 . Posibilidad de reconstruir imagenes tanto de targets individuales como
   de matrices de targets
 . Pruebas con datos simulados
 . Zero padding para enfocar mejor la imagen
@author: LUIS_DlCp
"""
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt
import sarPrm as sp # Modulo creado para definir los parametros del sistema
import drawFigures as dF # Modulo creado para graficar
import SARdata as Sd # Modulo creado para obtner el Historico de Fase teorico(Datos simulados)
import timeit
from mpl_toolkits.axes_grid1 import make_axes_locatable

#------------------LECTURA Y DEFINICIÓN DE PARÁMETROS--------------------
prm = sp.get_parameters()
c,fc,BW,Nf = prm['c'],prm['fc'],prm['BW'],prm['Nf']
Ls,Np,Ro,theta = prm['Ls'],prm['Np'],prm['Ro'],prm['theta']
Lx,Ly,dx,dy = prm['w'],prm['h'],prm['dw'],prm['dh'] # Dimensiones de la imagen
show = True

def get_SAR_data():
    """ Obtiene el histórico de fase ya sea simulado o real"""
    # Cálculo de parámetros
    It, Rt = sp.get_scalar_data2() # Magnitud y coordenadas del target respectivamente
    dp=Ls/(Np-1) # Paso del riel(m)
    df=BW/(Nf-1) # Paso en frecuencia del BW
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

    Ski = Sd.get_phaseH(prm,It,Rt) # Obtiene el historico de fase
    #np.save("RawData_Superman",Ski)
    #Ski = np.load("RawData_prueba1.npy")

    #-----------------GRÁFICA DEL HISTÓRICO DE FASE-----------------------
    dF.plotImage(Ski, x_min=fi, x_max=fs, y_min=-Ls/2, y_max=Ls/2,xlabel_name='Frecuencia(GHz)',
                 ylabel_name='Posición del riel(m)', title_name='Histórico de fase',unit_bar='', origin_n='upper')

    return {'Ski':Ski, 'df':df, 'fi':fi, 'rr_r':rr_r}

def FDBP_Algorithm(data1):
    """ Ejecuta el algoritmo Frequency Domain Back Projection"""
    # Lectura de parametros
    Ski = data1['Ski'].copy()
    df = data1['df']
    fi = data1['fi']
    rr_r = data1['rr_r']
    Lista_pos = np.linspace(-Ls/2, Ls/2, Np)

    # Grafica el perfil de rango para una posicion del riel(Posicion intermedia)
    if show:
        dF.rangeProfile(fi,BW,Ski[int(len(Ski)/2)],name_title='Perfil de rango\n(Magnitud)')

    # Hamming Windows
    S1 *= np.hamming(len(Ski[0])) # Multiply rows x hamming windows
    S1 = (S1.T*np.hamming(len(S1))).T # Multiply columns x hamming windows

    #----------PRIMERA ETAPA: 1D-IFFT respecto al eje de la 'f'-----------
    #---------------------------------------------------------------------
    # a) Agregar un factor de fase debido al delay de los cables
    S2 = S1*np.vander([np.exp(-1j*4*np.pi/c*Ro*df)]*Np,Nf,increasing=True) # Vandermonde matrix

    # b) Efectuar un zero padding en el eje de la 'f'
    zpad = 3*int(rr_r*(Nf-1)/dy)+1 # Dimension final despues del zero padding (depende del espaciado de la grilla, para tener mejor interpolacion), se le agrego el 3 de mas para una grilla igual a la resolucion y para identificar correctamente los targets
    col = int(zpad - len(S1[0])) # Length of zeros
    S3 = np.pad(S2, [[0, 0], [0, col]], 'constant', constant_values=0) # Aplica zero padding a ambos extremos de la matriz

    # c) Efectuar la IFFT
    S4 = np.fft.ifft(S3,axis=1)

    # GRAFICA
    if show:
        dkn = 4*np.pi*df/0.3
        rn = (np.arange(zpad))*2*np.pi/dkn/(zpad-1)
        fn = 20*np.log10(abs(S4[int(len(S4)/2)]))

        fig, ax= plt.subplots()
        ax.plot(rn,fn,'k')
        ax.set(xlabel='Rango(m)',ylabel='Intensidad(dB)', title='Perfil de rango')
        ax.grid()
        plt.show()

    #------------------SEGUNDA ETAPA: Interpolacion-----------------------
    # --------------------------------------------------------------------
    # a) Definicion del dominio de interpolacion
    dkn = 4*np.pi*df/0.3
    rn = (np.arange(zpad))*2*np.pi/dkn/(zpad-1)

    # b) Declaracion de las coordenadas de la imagen a reconstruir
    x_c=np.arange(-Lx/2,Lx/2+dx,dx)
    y_c=np.arange(0,Ly+dy,dy)
    r_c=np.array([(i,j) for j in y_c for i in x_c])

    # Funcion para calcular la distancia entre una posición del riel y todos las coordenadas de la imagen
    def distance_nk(r_n,x_k): # vector de coordenadas de la imagen, punto del riel "k"
        d=((r_n[:,0]-x_k)**2+(r_n[:,1])**2)**0.5
        return d

    # Funcion de interpolacion
    def interp(k,dist): # k: posicion del riel, vector de distancias entre 'n' y 'k'
        fr=sc.interp1d(rn,S4[k].real,kind='linear',bounds_error=False, fill_value=S4[k,-1].real) # Interpolacion al punto mas cercano
        fi=sc.interp1d(rn,S4[k].imag,kind='linear',bounds_error=False, fill_value=S4[k,-1].imag) # Fuera de la frontera se completa con el valor de la frontera
        return fr(dist) +1j*fi(dist)

    # c) Interpolacion y suma coherente
    S5=np.zeros(len(r_c),dtype=complex)
    for kr S5 range(Np):
        Rnk = distance_nk(r_c,Lista_pos[kr]) # Vector de distancias entre una posicion del riel y todos los puntos de la imagen
        Ke = np.exp(1j*4*np.pi/c*fi*(Rnk-Ro)) # Factor de fase
        Fnk = interp(kr,Rnk) # Valor interpolado en cada punto "n" de la grilla
        S5 += Fnk*Ke # Valor final en cada punto de la grilla
    S6 = S5/(Nf*Np)
    S7 = np.reshape(S6,(len(y_c),len(x_c)))

    # d) Normalizando la salida
    Im = S7/(np.sqrt(np.sum(abs(S5)**2)))

    return {'Im':Im}

def plot_image(data2):
    """ Grafica la magnitud de la imagen"""

    # a) Definicion y lectura de parametros
    Im = data2['Im']

    # b) Grafica final(magnitud)
    cmap="plasma"
    vmin = -100 #dB
    vmax = -20
    dF.plotImage(Im,cmap=cmap,xlabel_name='Azimut(m)',ylabel_name='Rango(m)', title_name='Resultado Algoritmo Back Projection',
                 x_min=-(Lx+dx)/2, x_max=(Lx+dx)/2, y_min=0-dy/2, y_max=Ly+dy/2,unit_bar='(dB)',log=True,vmin=vmin,
                 vmax=vmax)
    return Im

def main():
    plt.close('all') # Cerrar todas las figuras previas

    start_time = timeit.default_timer()
    datos = get_SAR_data() # Obtiene el historico de fase
    print("Tiempo de simulación(BP): ",timeit.default_timer() - start_time,"s")

    start_time = timeit.default_timer()
    d_p = FDBP_Algorithm(datos) # Implementa el algoritmo RMA
    plot_image(d_p) # Grafica de la magnitud
    print("Tiempo del procesamiento(BP): ",timeit.default_timer() - start_time,"s")

    return d_p['In']

if __name__ == '__main__':
  main()
