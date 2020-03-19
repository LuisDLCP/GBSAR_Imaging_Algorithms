# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 09:47:19 2018
-------------------------------------------------
      LIBRERÍA PARA GRAFICAR LAS IMAGENES
            CON DISTINTOS FORMATOS
-------------------------------------------------
@author: LUIS
"""
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# UNA SOLA GRAFICA DENTRO DE UNA SOLA VENTANA(Simple unit plot)
def simple_Uplot(x,y,title=None,axis_X=None,axis_Y=None):
    """
    Return a single image within a single window.

    Parameters
    ----------
    x : array
        "x" values of the function
    y : array
        "y" values of the function
    title : string
        Figure title name.
    axisX : string
        Image axis X name
    axisY : string
        Image axis Y name

    Returns
    -------
    A message confirmation

    """
    plt.figure()
    plt.plot(x,y)
    plt.xlabel(axis_X)
    plt.ylabel(axis_Y)
    plt.title(title)
    plt.grid()
    plt.show()

    return "Done!"

# GRAFICA DE VARIAS GRAFICAS(2) SEPARADAS DENTRO DE UNA SOLA VENTANA
def multiple_plot(x1,y1,x2,y2,title1=None,xlabel1=None,ylabel1=None,title2=None,xlabel2=None,ylabel2=None):
    """
    Return mutiple image within a single window.

    Parameters
    ----------
    x1,x2 : array
        "x" values of the function
    y1,y2 : array
        "y" values of the function
    title1,title2 : string
        Figure title name.
    xlabel1,xlabel2 : string
        Image axis X name
    ylabel1,ylabel2 : string
        Image axis Y name

    Returns
    -------
    A message confirmation

    """
    fig, ax= plt.subplots(2) # 2 subplots
    ax[0].plot(x1,y1)
    ax[0].set(xlabel=xlabel1,ylabel=ylabel1, title=title1)
    ax[0].grid()

    ax[1].plot(x2,y2)
    ax[1].set(xlabel=xlabel2,ylabel=ylabel2, title=title2)
    ax[1].grid()

    fig.tight_layout() # cuadra bien las imagenes

    return "Done!"


# GRAFICA DE VARIAS GRAFICAS JUNTAS EN UNA SOLA VENTANA

# GRAFICA DEL ESPECTRO DE FRECUENCIA(FFT){time --> frequency}, SOLO PARA FUNCIONES REALES
def plotFFT(x,y,x_unit="s",Xf_unit="Hz"):
    N=int(x.shape[0]) # Numero de puntos
    fs=1/(x[0]-x[1]) # Frecuencia de muestreo
    xf=np.fft.fftfreq(N, d=1/fs)
    yf=np.fft.fft(y)
    # Funcion para determinar el tipo de las unidades del eje x en el dominio del tiempo
    def f_xunits(x):
        return {
            "ms":1e3,
            "us":1e6,
            "ns":1e9
        }.get(x,1)
    # Funcion para determinar el tipo de unidades del eje x en el dominio de la frecuencia
    def f_Xfunits(x):
        return {
            "KHz":1e-3,
            "MHz":1e-6,
            "GHz":1e-9
        }.get(x,1)

    gs=gridspec.GridSpec(2,2)

    fig=plt.figure()
    # Grafica en el tiempo
    ax1=fig.add_subplot(gs[0,:])
    ax1.plot(x*f_xunits(x_unit),y)
    ax1.set(xlabel="Time"+"("+x_unit+")",ylabel="Magnitud", title="Funcion f(t)")
    ax1.grid()
    # Grafica en frecuencia(Magnitud)
    ax2=fig.add_subplot(gs[1,0])
    ax2.plot(np.fft.fftshift(xf)*f_Xfunits(Xf_unit),np.fft.fftshift(np.abs(yf)))
    ax2.set(xlabel="Frequency"+"("+Xf_unit+")",ylabel="Magnitud", title="Espectro de frecuencia\n(En Magnitud)")
    ax2.grid()
    # Grafica en frecuencia(Fase)
    ax3=fig.add_subplot(gs[1,1])
    ax3.plot(np.fft.fftshift(xf)*f_Xfunits(Xf_unit),np.fft.fftshift(np.angle(yf)))
    ax3.set(xlabel="Frequency"+"("+Xf_unit+")",ylabel="Angle(rad)", title="Espectro de frecuencia\n(En Fase)")
    ax3.grid()

    fig.tight_layout() # cuadra bien las imagenes
    return xf,yf


# GRAFICA DEL ESPECTRO DE FRECUENCIA(FFT){time --> frequency}, PARA FUNCIONES NO REALES(COMPLEJOS)
def plotFFTc(x,y,x_unit="s",Xf_unit="Hz"):
    N=int(x.shape[0]) # Numero de puntos
    fs=1/(x[0]-x[1]) # Frecuencia de muestreo
    xf=np.fft.fftfreq(N, d=1/fs)
    yf=np.fft.fft(y)
    # Funcion para determinar el tipo de las unidades del eje x en el dominio del tiempo
    def f_xunits(x):
        return {
            "ms":1e3,
            "us":1e6,
            "ns":1e9
        }.get(x,1)
    # Funcion para determinar el tipo de unidades del eje x en el dominio de la frecuencia
    def f_Xfunits(x):
        return {
            "KHz":1e-3,
            "MHz":1e-6,
            "GHz":1e-9
        }.get(x,1)

    gs=gridspec.GridSpec(2,2)

    fig=plt.figure()
    # Grafica en el tiempo(Magnitud)
    ax1=fig.add_subplot(gs[0,0])
    ax1.plot(x*f_xunits(x_unit),np.abs(y))
    ax1.set(xlabel="Time"+"("+x_unit+")",ylabel="Magnitud", title="Función f(t)\n(En Magnitud)")
    ax1.grid()
    # Grafica en el tiempo(Fase)
    ax1=fig.add_subplot(gs[0,1])
    ax1.plot(x*f_xunits(x_unit),np.angle(y))
    ax1.set(xlabel="Time"+"("+x_unit+")",ylabel="Angle(rad)", title="Función f(t)\n(En Fase)")
    ax1.grid()
    # Grafica en frecuencia(Magnitud)
    ax2=fig.add_subplot(gs[1,0])
    ax2.plot(np.fft.fftshift(xf)*f_Xfunits(Xf_unit),np.fft.fftshift(np.abs(yf)))
    ax2.set(xlabel="Frequency"+"("+Xf_unit+")",ylabel="Magnitud", title="Espectro de frecuencia\n(En Magnitud)")
    ax2.grid()
    # Grafica en frecuencia(Fase)
    ax3=fig.add_subplot(gs[1,1])
    ax3.plot(np.fft.fftshift(xf)*f_Xfunits(Xf_unit),np.fft.fftshift(np.angle(yf)))
    ax3.set(xlabel="Frequency"+"("+Xf_unit+")",ylabel="Angle(rad)", title="Espectro de frecuencia\n(En Fase)")
    ax3.grid()

    fig.tight_layout() # cuadra bien las imagenes
    return xf,yf

# GRAFICA DEL ESPECTRO DE FRECUENCIA(FFT){range(r) --> wavenumber(k)}, PARA FUNCIONES NO REALES(COMPLEJOS)
def plotFFTc_rk(x,y,x_unit="m",Xf_unit="1/m"):
    N=int(x.shape[0]) # Numero de puntos
    fs=1/(x[0]-x[1]) # Frecuencia de muestreo
    xf=np.linspace(-np.pi*fs,np.pi*fs,N,endpoint=True)
    yf=np.fft.fft(y)
    # Funcion para determinar el tipo de las unidades del eje x en el dominio del tiempo
    def f_xunits(x):
        return {
            "cm":1e2,
            "mm":1e3
            #"ns":1e9
        }.get(x,1)
    # Funcion para determinar el tipo de unidades del eje x en el dominio de la frecuencia
    def f_Xfunits(x):
        return {
            "KHz":1e-3,
            "MHz":1e-6,
            "GHz":1e-9
        }.get(x,1)

    gs=gridspec.GridSpec(2,2)

    fig=plt.figure()
    # Grafica en el dominio del rango(Magnitud)
    ax1=fig.add_subplot(gs[0,0])
    ax1.plot(x*f_xunits(x_unit),np.abs(y))
    ax1.set(xlabel="Rango"+"("+x_unit+")",ylabel="Magnitud", title="Función f(r)\n(En Magnitud)")
    ax1.grid()
    # Grafica en el dominio del rango(Fase)
    ax1=fig.add_subplot(gs[0,1])
    ax1.plot(x*f_xunits(x_unit),np.angle(y))
    ax1.set(xlabel="Rango"+"("+x_unit+")",ylabel="Angle(rad)", title="Función f(r)\n(En Fase)")
    ax1.grid()
    # Grafica en el dominio del numero de onda k(Magnitud)
    ax2=fig.add_subplot(gs[1,0])
    ax2.plot(xf*f_Xfunits(Xf_unit),np.fft.fftshift(np.abs(yf)))
    ax2.set(xlabel="Número de onda k"+"("+Xf_unit+")",ylabel="Magnitud", title="Espectro de frecuencia\n(En Magnitud)")
    ax2.grid()
    # Grafica en frecuencia del numero de onda k(Fase)
    ax3=fig.add_subplot(gs[1,1])
    ax3.plot(xf*f_Xfunits(Xf_unit),np.fft.fftshift(np.angle(yf)))
    ax3.set(xlabel="Numero de onda k"+"("+Xf_unit+")",ylabel="Angle(rad)", title="Espectro de frecuencia\n(En Fase)")
    ax3.grid()

    fig.tight_layout() # cuadra bien las imagenes
    return xf,yf

# Grafica del range profile, la data de frecuencias tiene q estar en GHz
def rangeProfile(fi,BW,dataf,name_title='Range Profile',file=None): # frecuencia inicial(GHz),Bandwidth, datos en el dominio de la frecuencia
    Nf=len(dataf) # Numero de datos(frecuencias)
    #dr=0.3/(2*BW) # Resolucion(m)
    #df=BW/(Nf-1)
    #dk=4*np.pi/0.3*df # El 4 x el ida y vuelta

    #  Interpolacion
    Nf_n=1*Nf
    f = np.linspace(fi,fi+BW,Nf) # frecuencias originales
    f_n = np.linspace(fi,fi+BW,Nf_n) # frecuencias de interpolación(con el doble de Longitud)

    DATA_it=np.zeros(len(f_n))
    data_nr = sc.interpolate.interp1d(f, dataf, bounds_error=False, fill_value=0)
    data_ni = sc.interpolate.interp1d(f, dataf, bounds_error=False, fill_value=0)
    DATA_it = data_nr(f_n)+1j*data_ni(f_n)

    # Zero padding
    zpad= 4096 # Final size of dataf
    rows = int(zpad - Nf_n) # Size of zeros
    data_n = np.pad(DATA_it, [0,rows], 'constant', constant_values=0)

    # 1D-IFFT
    F1=abs(np.fft.ifft(data_n)) #np.fft.fftshift(np.fft.ifft(data_n)))) # Values in range domain
    # Calculo del rango
    dkn=4*np.pi*(f_n[1]-f_n[0])/0.3
    R=(np.arange(zpad))*2*np.pi/dkn/zpad #(np.arange(zpad)-zpad/2)*2*np.pi/dk/zpad #np.linspace(-dr*Nf/2,dr*Nf/2,zpad) #np.arange(0,dr*Nf,dr) # range vector
    # R2 = R[270:2430]
    # F2 = F1[270:2430]
    # Plot
    fig, ax= plt.subplots()
    ax.plot(R,F1,'k')
    ax.set(xlabel='Range (m)',ylabel='Amplitude(dBsm)', title=name_title)
    ax.set_xlim([R.min(),R.max()])
    ax.set_ylim([F1.min(),F1.max()])
    ax.grid()
    plt.show()
    if file!=None:
        fig.savefig(file)

    return "ok"

# Grafica del crossrange profile, la data de posicion
def crangeProfile(da,datap): # resolucion en azimuth, datos en el dominio del rango
    Np=len(datap) # Numero de datos(posiciones)
    R=np.arange(0,da*Np,da) # range vector

    F1=20*np.log10(abs((np.fft.ifft(np.fft.ifftshift(datap))))) # Values in range domain

    fig, ax= plt.subplots()
    ax.plot(R,F1,'k')
    ax.set(xlabel='Cross-Range (m)',ylabel='Amplitude(dBsm)', title='Cross-Range Profile')# of Ro='+str([Rx,Ry]))
    ax.set_xlim([R.min(),R.max()])
    ax.set_ylim([F1.min(),F1.max()])
    ax.grid()
    plt.show()

    return "ok"

# Grafica de una imagen en Magnitud y Fase
def plotImage(data, x_min=None, x_max=None, y_min=None, y_max=None, xlabel_name=None, ylabel_name=None, title_name=None, unit_bar='Magnitude', origin_n='lower', log=False):
    """
    Parameters
    ----------
    data : 2D-matrix
        Data a ser mostrado como figura
    x_min : float
        Valor mínimo del eje "x"
    x_max : float
        Valor máximo del eje "x"
    y_min : float
        Valor mínimo del eje "y"
    y_max : float
        Valor máximo del eje "y"

    Returns
    -------
    A message confirmation.
    """
    cmap="hot"
    r_data = abs(data)
    if log: r_data = 20*np.log10(r_data)

    fig, ax = plt.subplots(1,2)#,sharex=True)
    # Magnitude
    im1=ax[0].imshow(r_data,cmap,origin=origin_n,extent=[x_min, x_max, y_min, y_max], aspect='auto')#vmin=-600,vmax=-100)
    ax[0].set(xlabel=xlabel_name, ylabel=ylabel_name, title="(Magnitude)") # Origin 'upper': esquina superior izquierda; 'lower': esquina inferior izquierda
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1) # pad es el espaciado con la grafica principal
    plt.colorbar(im1,cax=cax1,label=unit_bar,extend='both')
    ax[0].grid()
    # Phase
    im2=ax[1].imshow(np.angle(data), cmap, origin=origin_n, extent=[x_min, x_max, y_min, y_max], aspect='auto')#vmin=-600,vmax=-100)
    ax[1].set(xlabel=xlabel_name,ylabel=ylabel_name, title="(Phase)")
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes("right", size="5%", pad=0.1) # pad es el espaciado con la grafica principal
    ax[1].grid()
    plt.colorbar(im2,cax=cax2,label='Angle(rad)',extend='both')
    fig.suptitle(title_name)
    fig.subplots_adjust(left=0.065, right=0.95, wspace=0.3)
    #fig.tight_layout() # cuadra bien las imagenes
    return 'Ok'
