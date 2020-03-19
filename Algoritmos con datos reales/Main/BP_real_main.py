#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:14:46 2018
-------------------------------------------------------------
                ALGORITMO FD-BACK PROJECTION
-------------------------------------------------------------
Descripción:
 . El presente algoritmo asume las siguientes consideraciones:
    - Riel en el eje X.
 . Posibilidad de reconstruir imagenes tanto de targets individuales como
   de matrices de targets.
 . Datos experimentales.
 . Para mayor información revisar los papers: "Terrain Mapping by Ground-Based Interferometric Radar"
                                                by Massimiliano Pieraccini.

@author: LUIS_DlCp
"""
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt
import sarPrm as sp # Library created to define sistem parameters
import drawFigures as dF # Library created to draw FFT functions
import timeit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py as hp
import os

os.chdir(os.path.dirname(__file__))

#-----------------------LECTURA DE PARÁMETROS-----------------------------
"""f = hp.File('dset_33.hdf5','r')
dset = f['sar_dataset']

prm = sp.get_parameters2(dset)
c,fc,BW,Nf = prm['c'],prm['fc'],prm['BW'],prm['Nf']
Ls,Np,Ro,theta = prm['Ls'],prm['Np'],prm['Ro'],prm['theta']
Lx,Ly,dx,dy,yi = prm['w'],prm['h'],prm['dw'],prm['dh'],prm['hi'] # Dimensiones de la imagen
dp2 = prm['dx']"""

def get_SAR_data():
    """ Obtiene el histórico de fase ya sea simulado o real"""
    # Cálculo de parámetros
    dp = Ls/(Np-1) # Paso del riel(m)
    df = BW/(Nf-1) # Paso en frecuencia del BW
    fi = fc-BW/2 # Frecuencia inferior(GHz)
    fs = fc+BW/2 # Frecuencia superior(GHz)

    # Rango máximo
    R_max=Nf*c/(2*BW)
    # Paso del riel máximo
    dp_max=c/(fc*4*np.sin(theta*np.pi/180)) # Theta en grados sexagesimales

    # Cálculo de las resoluciones
    rr_r = c/(2*BW) # Resolución en rango
    rr_a = c/(2*Ls*fc) # Resolución en azimuth

    #-----------------VERIFICACIÓN DE CONDICIONES------------------------
    print("------------------------------------------------------")
    print("--------------INFORMACIÓN IMPORTANTE------------------")
    print("------------------------------------------------------")
    print("- Resolución en rango(m) : ", rr_r)
    print("- Resolución en azimuth(rad): ", rr_a)
    print("------------------------------------------------------")
    print("- Rango máximo permitido(m): ", R_max)
    print("------------------------------------------------------")
    print("______¿Se cumplen las siguientes condiciones?_________")
    print("Rango máximo del target <= rango máximo?: ", R_max<=R_max) # Ponerle un try-except
    print("Paso del riel <= paso máximo?: ", dp<=dp_max) # Evita el aliasing en el eje de azimuth
    print("------------------------------------------------------")

    #----------------OBTENCIÓN DEL HISTÓRICO DE FASE----------------------
    Sr_f = np.array(list(dset))

    #-----------------GRÁFICA DEL HISTÓRICO DE FASE-----------------------
    dF.plotImage(Sr_f, x_min=fi, x_max=fs, y_min=-Ls/2, y_max=Ls/2,xlabel_name='Frecuency(GHz)',
                 ylabel_name='Riel Position(m)', title_name='Histórico de fase',unit_bar='dBu', origin_n='upper')

    return {'Sr_f':Sr_f, 'df':df, 'fi':fi, 'rr_r':rr_r}

def FDBP_Algorithm(data1):
    """ Ejecuta el algoritmo Frequency Domain Back Projection"""
    # Lectura de parametros
    Sr_f = data1['Sr_f'].copy()
    df = data1['df']
    fi = data1['fi']
    rr_r = data1['rr_r']
    Lista_pos = np.linspace(-Ls/2, Ls/2, Np)

    # Grafica el perfil de rango para una posicion del riel(Posicion intermedia)
    if show:
        dF.rangeProfile(fi,BW,Sr_f[int(len(Sr_f)/2)],name_title='Perfil de rango\n(Magnitud)')

    # Hanning Windows
    Sr_f *= np.hanning(len(Sr_f[0])) # Multiply rows x hanning windows
    Sr_f = (Sr_f.T*np.hanning(len(Sr_f))).T # Multiply columns x hanning windows

    #----------PRIMERA ETAPA: 1D-IFFT respecto al eje de la 'f'-----------
    #---------------------------------------------------------------------
    # a) Agregar un factor de fase debido al delay de los cables
    E1 = Sr_f*np.vander([np.exp(-1j*4*np.pi/c*Ro*df)]*Np,Nf,increasing=True) # Vandermonde matrix

    # b) Efectuar un zero padding en el eje de la 'f'
    zpad = 3*int(rr_r*(Nf-1)/dy)+1 # Dimension final despues del zero padding (depende del espaciado de la grilla, para tener mejor interpolacion), se le agrego el 3 de mas para una grilla igual a la resolucion y para identificar correctamente los targets
    col = int(zpad - len(Sr_f[0])) # Length of zeros
    E1 = np.pad(E1, [[0, 0], [0, col]], 'constant', constant_values=0) # Aplica zero padding a ambos extremos de la matriz

    # c) Efectuar la IFFT
    E2 = np.fft.ifft(E1,axis=1)

    # GRAFICA
    if show:
        dkn = 4*np.pi*df/0.3
        rn = (np.arange(zpad))*2*np.pi/dkn/(zpad-1)
        fn = 20*np.log10(abs(E2[int(len(E2)/2)]))

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
    x_c = np.arange(-Lx/2,Lx/2+dx,dx)
    y_c = np.arange(yi,yi+Ly+dy,dy)
    r_c=np.array([(i,j) for j in y_c for i in x_c])

    # Funcion para calcular la distancia entre una posición del riel y todos las coordenadas de la imagen
    def distance_nk(r_n,x_k): # vector de coordenadas de la imagen, punto del riel "k"
        d=((r_n[:,0]-x_k)**2+(r_n[:,1])**2)**0.5
        return d

    # Funcion de interpolacion
    def interp(k,dist): # k: posicion del riel, vector de distancias entre 'n' y 'k'
        fr=sc.interp1d(rn,E2[k].real,kind='linear',bounds_error=False, fill_value=E2[k,-1].real) # Interpolacion al punto mas cercano
        fi=sc.interp1d(rn,E2[k].imag,kind='linear',bounds_error=False, fill_value=E2[k,-1].imag) # Fuera de la frontera se completa con el valor de la frontera
        return fr(dist) +1j*fi(dist)

    # c) Interpolacion y suma coherente
    In=np.zeros(len(r_c),dtype=complex)
    for kr in range(Np):
        Rnk = distance_nk(r_c,Lista_pos[kr]) # Vector de distancias entre una posicion del riel y todos los puntos de la imagen
        Ke = np.exp(1j*4*np.pi/c*fi*(Rnk-Ro)) # Factor de fase
        Fnk = interp(kr,Rnk) # Valor interpolado en cada punto "n" de la grilla
        In += Fnk*Ke # Valor final en cada punto de la grilla
    In /= Nf*Np
    In = np.reshape(In,(len(y_c),len(x_c)))

    # d) Normalizando la salida
    In /= np.sqrt(np.sum(abs(In)**2))

    return {'In':In}

def plot_image(data2):
    """ Grafica la magnitud de la imagen"""
    # a) Definicion y lectura de parametros
    Im = data2['In'].copy()
    title = data2['file_name']
    title = title[:-5]
    title_name='Imagen Final(BP)\n'+title
    direction ='Imagen Final_BP_'+title

    # b) Grafica final(magnitud)
    cmap = "plasma"
    vmin = np.amin(20*np.log10(abs(Im)))+53
    vmax = np.amax(20*np.log10(abs(Im)))
    fig, ax = plt.subplots()
    im=ax.imshow(20*np.log10(abs(Im)), cmap, origin='lower', aspect='equal', extent=[-Lx/2, Lx/2, yi, yi+Ly],vmin=vmin,vmax=vmax)
    ax.set(xlabel='Azimut(m)',ylabel='Rango(m)', title=title_name)
    ax.grid()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1) # pad es el espaciado con la grafica principal
    plt.colorbar(im,cax=cax,label='Reflectividad(dB)',extend='both')

    fig.savefig(os.getcwd()+"/Results/Imagenes_finales_reconstruidas_BP/"+direction,orientation='landscape')

    return {'Im':Im, 'x_min':-Lx/2, 'x_max':Lx/2, 'y_min':yi, 'y_max':yi+Ly}

def main(dset_name):
    plt.close('all') # Cerrar todas las figuras previas

    # Se declaran y cargan variables
    f = hp.File(os.getcwd()+"/Data_set/"+dset_name,'r')
    global dset
    dset = f['sar_dataset']
    prm = sp.get_parameters2(dset)

    global c,fc,BW,Nf,Ls,Np,Ro,theta,Lx,Ly,dx,dy,yi,dp2,show

    c,fc,BW,Nf = prm['c'],prm['fc'],prm['BW'],prm['Nf']
    Ls,Np,Ro,theta = prm['Ls'],prm['Np'],prm['Ro'],prm['theta']
    Lx,Ly,dx,dy,yi = prm['w'],prm['h'],prm['dw'],prm['dh'],prm['hi'] # Dimensiones de la imagen
    dp2 = prm['dx']
    date = prm['date'].decode('utf-8') # string data

    show = False

    # Obtencion del Raw Data
    start_time = timeit.default_timer()
    datos = get_SAR_data() # Obtiene el historico de fase
    print("Tiempo de simulación: ",timeit.default_timer() - start_time,"s")

    # Procesamiento BP
    start_time = timeit.default_timer()
    d_p = FDBP_Algorithm(datos) # Implementa el algoritmo FDBP
    print("Tiempo del procesamiento: ",timeit.default_timer() - start_time,"s")

    # Graficas
    d_p['file_name'] = dset_name

    IF = plot_image(d_p)
    IF['date'] = date # Se anexa al diccionario

    return IF

if __name__ == '__main__':
    test = main("dset_60.hdf5")
