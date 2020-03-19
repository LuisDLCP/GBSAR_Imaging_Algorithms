# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:10:41 2019
-------------------------------------------------------------
                ALGORITMO RANGE-MIGRATION
-------------------------------------------------------------
Descripción:
 . El presente algoritmo asume las siguientes consideraciones:
    - Riel en el eje X.
 . Posibilidad de reconstruir imagenes tanto de targets individuales como
   de matrices de targets.
 . Datos reales.
 . Grilla uniforme e igual a la resolucion 
 . Con Decimación
 . Para mayor información revisar los libros: Carrara, Charvat.
 . Se optimiza el codigo para disminuir el tp: decimacion, espectro recortado despues del FFT
 . Algoritmo final optimizado
@author: LUIS_DlCp
"""
import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt
import sarPrm as sp # Library created to define sistem parameters
import drawFigures as dF # Library created to draw FFT functions
import timeit
import h5py as hp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os 

os.chdir(os.path.dirname(__file__))

#-----------------------LECTURA DE PARÁMETROS-----------------------------
"""f = hp.File('dset_33.hdf5','r')
dset = f['sar_dataset']

prm = sp.get_parameters2(dset) 
c,fc,BW,Nf = prm['c'],prm['fc'],prm['BW'],prm['Nf']
Ls,Np,Ro,theta = prm['Ls'],prm['Np'],prm['Ro'],prm['theta']
W,H,dx,dy,yi = prm['w'],prm['h'],prm['dw'],prm['dh'],prm['hi'] # Dimensiones de la imagen
dp2 = prm['dx']
Npx,Npy = int(np.ceil(W/dx)),int(np.ceil(H/dy)) # Numero de pixeles finales si hay decimacion
show = False"""

def get_SAR_data():
    """ Obtiene el histórico de fase ya sea simulado o real"""
    # Cálculo de parámetros
    dp = Ls/(Np-1) # Paso del riel(m)
    fi = fc-BW/2 # Frecuencia inferior(GHz)
    fs = fc+BW/2 # Frecuencia superior(GHz)
    
    # Rango máximo
    R_max=Nf*c/(2*BW)
    # Paso del riel máximo
    dp_max=c/(fc*4*np.sin(theta*np.pi/180)) # Theta en grados sexagesimales
    
    # Cálculo de las resoluciones
    rr_r = c/(2*BW) # Resolución en rango
    rr_a = (c/(2*Ls*fc)) # Resolución en azimuth
    # Nf2 = np.ceil(H/rr_r) # Numero de frecuencias final 
    
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
    if show:
        dF.plotImage(Sr_f, x_min=fi, x_max=fs, y_min=-Ls/2, y_max=Ls/2,xlabel_name='Frecuency(GHz)', 
                     ylabel_name='Riel Position(m)', title_name='Histórico de fase',unit_bar='dBu', origin_n='upper')
    
    return {'Sr_f':Sr_f, 'dp':dp, 'fi':fi, 'fs':fs, 'R_max':R_max}

def RMA_Algorithm(data1):
    """ Ejecuta el algoritmo RMA"""
    # Lectura de parametros
    global dec_x, dec_y
    Sr_f = data1['Sr_f'].copy()
    dp = data1['dp']
    fi = data1['fi']
    fs = data1['fs']
    R_max = data1['R_max']
    Lista_f = np.linspace(fi, fs, Nf) # Vector de frecuencias

    # Grafica el perfil de rango para una posicion del riel(Posicion '0')
    if show:
        dF.rangeProfile(fi,BW,Sr_f[int(len(Sr_f)/2)],name_title='Perfil de rango \n(Al principio)')

    # Hanning Windows
    Sr_f *= np.hanning(len(Sr_f[0])) # Multiply rows x hanning windows 
    Sr_f = (Sr_f.T*np.hanning(len(Sr_f))).T # Multiply columns x hanning windows
        
    #----------------ETAPA PREVIA: Range Compression----------------------
    #                       s(x,w) --> s(x,r)
    #---------------------------------------------------------------------
    if show:
        # a) IFFT (eje de rango)
        ifft_len = 10*len(Sr_f[0])
        S0 = np.fft.ifft(Sr_f, ifft_len, axis=1) # axis 0 representa las filas(axis 1 a las columnas), lo cual quiere decir q se sumaran todas las filas manteniendo constante las columnas
        # S0 = np.fft.fftshift(S0,axes=(1)) # to center spectrum
        
        dF.plotImage(S0.T, x_min=-Ls/2, x_max=Ls/2, y_min=0, y_max=R_max,xlabel_name='Azimut(m)',
                     ylabel_name='Rango(m)', title_name='Compresión en Rango',unit_bar='',
                     origin_n='lower')
    
    #----------PRIMERA ETAPA: FFT respecto al eje del riel 'x'----------
    #                       s(x,w) --> s(kx,kr)
    #                           Sr_f -> S1
    #---------------------------------------------------------------------

    # a) Zero padding en cross range
    zpad = int(W/dp) # Initial length of padded signal
    dec_x = int(np.ceil(zpad/Npx)) # Hallando el factor de decimacion en x
    if zpad > len(Sr_f):
        # Reajuste del Nª de elementos para la decimacion en x
        if zpad % Npx != 0:
            ad = dec_x*Npx-zpad # To have a hole multiple of dec
            zpad += ad # Final length of padded signal
        rows = int(zpad - len(Sr_f)) # Length of zeros
    else: rows = 0
    S1 = np.pad(Sr_f, [[0, rows], [0, 0]], 'constant', constant_values=0) # Aplica zero padding a ambos extremos de la matriz
    
    # b) FFT (eje de azimuth)
    S1 = np.fft.fft(S1,axis=0) # axis 0 representa las filas(axis 1 a las columnas), lo cual quiere decir q se sumaran todas las filas manteniendo constante las columnas
    S1 = np.fft.fftshift(S1,axes=(0)) # to center spectrum
    
    # c) Grafica despues del FFT
    kr=4*np.pi*Lista_f/c # Numeros de onda dependientes de la frecuencia
    kx=np.fft.fftfreq(len(S1))*2*np.pi/dp
    kx=np.fft.fftshift(kx)
    if show:
        dF.plotImage(S1.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=kr.min()-(kr[1]-kr[0])/2, y_max=kr.max()+(kr[1]-kr[0])/2,xlabel_name='kx(1/m)',
                     ylabel_name='kr(1/m)', title_name='1D-FFT\n(Dirección de azimut)', origin_n='lower',unit_bar='dB',log=True)

    # d) Recortando las zonas donde no hay mucha señal(-60dB del maximo) 
    # Hallando las zonas que cumplen la condicion de los -60dB
    if show:
        S1_test = S1.copy()
        mask = 20*np.log10(S1_test)>=20*np.log10(S1_test.max())-60 
        S1_test[~mask] = 0
        
        dF.plotImage(S1_test.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=kr.min()-(kr[1]-kr[0])/2, y_max=kr.max()+(kr[1]-kr[0])/2,xlabel_name='kx(1/m)',
                     ylabel_name='kr(1/m)', title_name='1D-FFT\n(Test de magnitud)', origin_n='lower',unit_bar='dB',log=True)
    
    """# Función para hallar ni a partir de yi
    def set_y(yi,Nt,yo,yf): # yi deseado, numero de valores de y, y inicial, y final
        Yit = (Nt-1)*(yi-yo)/(yf-yo) # float
        ni = int(np.floor(round(Yit,5)))
        return ni

    #ky_inf = set_y(400,len(ky_it),ky_it[0],ky_it[-1])
    #ky_sup = set_y(50,len(ky_it),ky_it[0],ky_it[-1])
    kx_inf = set_y(-400,len(kx),kx[0],kx[-1]) # Desde kx=-400
    kx_sup = set_y(400,len(kx),kx[0],kx[-1]) # Hasta kx=400
    
    kx = kx[kx_inf+2:kx_sup] # Correccion manual para la decimacion en x
    #ky_it = ky_it[:] 
    S1 = S1[kx_inf+2:kx_sup,:] 
    
    # Reajusta el Nº de elementos para la decimacion en "x"
    dec_x = int(np.ceil(len(kx)/Npx)) # Hallando el factor de decimacion en x
    adx = Npx*dec_x-len(kx) # Cantidad de valores a agregar
    if show:    
        dF.plotImage(S1.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=kr.min()-(kr[1]-kr[0])/2, y_max=kr.max()+(kr[1]-kr[0])/2,xlabel_name='kx(1/m)',
                     ylabel_name='kr(1/m)', title_name='1D-FFT\n(Dirección de azimut - Recortado)', origin_n='lower',unit_bar='dB',log=True)
    """
    #----------------SEGUNDA ETAPA: Matched Filter----------------------
    #                  s(kx,kr) --> sm(kx,kr)
    #                       S1  ->  S1_m
    #-------------------------------------------------------------------

    # a) Matched Filter
    krr, kxx= np.meshgrid(kr,kx)
    ky = np.sqrt(krr**2-kxx**2)
    
    Xc1 = - Rmax/2 #(yi+H/2) # Factor de correccion debido a la FFT en "y"
    Xc2 = -Ls/2 # Factor de correccion debido a la FFT en "x"
    Xc3 = 0 #-4 
    
    mask = np.isnan(ky) # Obtiene las posiciones de los números complejos

    phi_m = -kxx*(Xc2+Xc3)-Xc1*ky#+Xc1*krr
    S1 *= np.exp(1j*phi_m) # matrix
    S1[mask] = 0+0j
    
    # c) Grafica despues del Matched Filter
    if show:
        dF.plotImage(S1.T, x_min=kx.min()-(kx[1]-kx[0])/2, x_max=kx.max()+(kx[1]-kx[0])/2, y_min=kr.min()-(kr[1]-kr[0])/2, y_max=kr.max()+(kr[1]-kr[0])/2, xlabel_name='kx(1/m)',
                     ylabel_name='kr(1/m)', title_name='Matched Filter', origin_n='lower',unit_bar='')

    # d) Range and azimuth compression
    if show:
        S1_n = S1/(np.sqrt(np.sum(abs(S1)**2)))
        ifft_len = np.array([len(S1_n),10*len(S1_n[0])])
        S1_1 = np.fft.fftshift(S1_n, axes=(0))
        S1_1 = np.fft.ifft2(S1_1,ifft_len)
        S1_1 = np.fft.fftshift(S1_1, axes=(0,1))
        vmin=0
        vmax=5
        dF.plotImage(S1_1.T, x_min=-Ls/2, x_max=Ls/2, y_min=0, y_max=R_max,xlabel_name='Azimut(m)',
                     ylabel_name='Rango(m)', title_name='Compresión en Rango y Azimut\n(Despues del Matched Filter)',
                     origin_n='lower',log=True,unit_bar='(dB)')#,vmin=vmin,vmax=vmax)
    
    #----------------TERCERA ETAPA: Stolt Interpolation-------------------
    #                  F(kxmn,kymn) --> F(kx,ky)
    #                       S1_m  -->  S2
    #---------------------------------------------------------------------
    # a) Redefinicion de los ejes
    dk_r = (kr[1]-kr[0])
    
    # b) Interpolación STOLT
    ky_min = np.min(ky[~(np.isnan(ky))])
    ky_max = np.max(ky[~(np.isnan(ky))])
    ky_it = np.arange(ky_min,ky_max,dk_r) # Puntos de interpolacion en el eje ky
    
    # Reajusta el Nº de elementos para la decimacion en "y"
    dec_y = int(np.ceil(len(ky_it)/Npy)) # Hallando el factor de decimacion en y
    if len(ky_it) % Npy != 0:
        ady = dec_y*Npy-len(ky_it)
        ky_it = np.append(ky_it,[ky_it[-1]+dk_r*(i+1) for i in range(ady)])
        
    # Interpolacion y decimacion optimizada
    indx = int(len(S1)/dec_x) # Longitud despues de la decimacion en el eje X 
    indy = int(len(ky_it)/(dec_y)) # Longitud despues de la decimacion en el eje Y
    S2 = np.zeros((indx,indy), dtype=complex) # Matriz interpolada y decimada

    ky_it2 = np.reshape(ky_it,(dec_y,int(len(ky_it)/dec_y)))
    
    for i in range(dec_x):
        for k in range(indx):
            # Se redefinen ky y S1 eliminando los terminos nan
            ky_aux2 = ky[k+indx*i]
            S1_aux2 = S1[k+indx*i]
            S1_aux2 = S1_aux2[~np.isnan(ky_aux2)]            
            ky_aux2 = ky_aux2[~np.isnan(ky_aux2)]
            
            if ky_aux2.size>1:
                interp_fn1 = sc.interpolate.interp1d(ky_aux2, S1_aux2.real, bounds_error=False, fill_value=0,kind='cubic')
                interp_fn2 = sc.interpolate.interp1d(ky_aux2, S1_aux2.imag, bounds_error=False, fill_value=0,kind='cubic')
    
                for z in range(dec_y):
                    if ky_it2[z].max()>=ky_aux2.min() and ky_it2[z].min()<=ky_aux2.max():
                        if ky_it2[z].min()<=ky_aux2.min(): 
                            start = z
                        if ky_it2[z].max()>=ky_aux2.max(): 
                            end = z
                            break
                ky_it3 = np.reshape(ky_it2[start:end+1,:],-1)
                aux = interp_fn1(ky_it3)+1j*interp_fn2(ky_it3)
                S2[k] += aux.reshape((int(len(ky_it3)/indy),indy)).sum(axis=0)
                
    S2 /= dec_x*dec_y
    
    # Nuevos valores del eje "kx"
    Nkx = int(len(kx)/dec_x)
    kx_n = (np.arange(Nkx)-Nkx/2)*(kx[1]-kx[0])

    # Nuevos valores del eje "ky"
    Nky = int(len(ky_it)/dec_y)
    ky_n = np.arange(Nky)*(ky_it[1]-ky_it[0])+ky_it.min()


    # d) Grafica despues de la interpolación
    if show:
        dF.plotImage(S2.T, x_min=kx_n.min(), x_max=kx_n.max(), y_min=ky_n.min(), y_max=ky_n.max(), xlabel_name='kx(1/m)',
                 ylabel_name='ky(1/m)', title_name='Interpolación STOLT\n(Decimado)', unit_bar='(dB)', log=True)
    
    #--------CUARTA ETAPA_previa: 1D-IFFT(range compression)--------------------
    #                  F(kx,ky) --> f(kx,y)
    #----------------------------------------------------------
    if show:
        # a) 1D-IFFT
        ifft_len = 1*np.array(np.shape(S2))
        S3_1 = np.fft.ifft(S2,axis=1)
        S3_1 = np.fft.fftshift(S3_1, axes=(1))
        
        # b) Definicion de parametros para las graficas
        #Nx = ifft_len[0]
        Ny = ifft_len[1]
        #dk_x = (kx_n[1]-kx_n[0])
        dk_y = (ky_n[1]-ky_n[0])
        
        # x = (np.arange(Nx)-Nx/2)*(2*np.pi/dk_x/Nx)
        y = (np.arange(Ny)-(Ny-1)/2)*(2*np.pi/dk_y/(Ny-1))-Xc1
        
        # d) Grafica despues de la interpolación
        dF.plotImage(S3_1.T, x_min=kx_n.min(), x_max=kx_n.max(), y_min=y.min(), y_max=y.max(), xlabel_name='kx(1/m)',
                     ylabel_name='y(m)', title_name='Compresión en rango', unit_bar='')
    
    #-----------------CUARTA ETAPA: 2D-IFFT--------------------
    #                  F(kx,ky) --> f(x,y)
    #                     S2  -->  Sf
    #----------------------------------------------------------
    # a) 2D-IFFT
    ifft_len = np.array([1*len(S2),1*len(S2[0])])#np.array(np.shape(S3)) #: Uncomment si no quiere interpolar  
    S3 = np.fft.ifftshift(S2, axes=(0)) # De-centra el eje kx
    S3 = np.fft.ifft2(S3, ifft_len)
    S3 = np.fft.fftshift(S3, axes=(0,1)) # Centra en el origen
    
    # b) Definicion de parametros para las graficas
    Nx = ifft_len[0]
    Ny = ifft_len[1]
    dk_x = (kx_n[1]-kx_n[0])
    dk_y = (ky_n[1]-ky_n[0])
    
    x = np.linspace(-np.pi/dk_x,np.pi/dk_x,Nx)-Xc3
    y = np.linspace(-np.pi/dk_y,np.pi/dk_y,Ny)-Xc1
    
    # Compensacion de fase 
    y2,x2 = np.meshgrid(y,x)
    kx_c = 0 #10*np.pi #(kx_n[-1]-kx_n[0])/2 # considerar solo cuando no se hace fftshift en x
    ky_c = (kr.max()-kr.min())/2 #4*np.pi + 1.37/5 #ky_n[0]-2*np.pi/0.55-2*np.pi/2#ky_n[int(len(ky_n)/2)] #4*np.pi
    kyo = 0 #-0.57
    S3 *= np.exp(-1j*kx_c*x2+1j*(ky_c*y2+kyo)) 
    
    # Normalizando la salida 
    S3 /= np.sqrt(np.sum(abs(S3)**2))
    
    if show:
        # GRAFICA DEL PERFIL DE RANGO(Magnitud) 
        fig, ax = plt.subplots()
        ax.plot(y, 20*np.log10(abs(S3[int(len(S3)/2)])),'k')
        ax.set(xlabel='Rango(m)',ylabel='Intensidad(dB)', title='Perfil de rango\n(Después de la Interpolación)')
        ax.grid()
        plt.show()
        # GRAFICA DEL PERFIL DE FASE(Eje del rango) 
        fig, ax = plt.subplots()
        ax.plot(y, np.angle(S3[int(len(S3)/2)]),'k')
        ax.set(xlabel='Rango(m)',ylabel='Fase(rad)', title='Perfil de fase\n(Después de la Interpolación)')
        ax.grid()
        plt.show()
        # GRAFICA DEL PERFIL DE Fase(Eje de cross range)
        fig, ax = plt.subplots()
        ax.plot(x, np.angle(S3.T[int(len(S3[0])/2)]),'k')
        ax.set(xlabel='Rango(m)',ylabel='Fase(rad)', title='Perfil de fase - cr\n(Después de la Interpolación)')
        ax.grid()
        plt.show()
    
    return {'Sf':S3,'x':x,'y':y}

def plot_image(data2):
    """ Grafica la magnitud de la imagen"""
    # a) Definicion y lectura de parametros 
    Sf = data2['Sf']
    x = data2['x']
    y = data2['y']
    #R_max = data2['R_max']
    Ny = len(Sf[0])
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    
    # b) Truncacion de la imagen(Eje Y) 
    # Función para hallar ni a partir de yi
    def set_xy(yi):
        return (Ny-1)*(yi-y[0])/(y[-1]-y[0]) # float

    y1,y2 = yi,yi+H #y.min(),y.max() #300,950 #max(0,y.min()),min(R_max,y.max()) # Nuevos valores limites de y=[y1,y2]
    n1 = int(np.floor(round(set_xy(y1),5))) # n1 = f(y1)
    n2 = int(np.ceil(round(set_xy(y2),5))) # n2 = f(y2)
    # Se uso primero round, floor y ceil, para asegurarse de q n1, n2 no salga de los limites permitidos
    Sf_n = Sf.T[n1:n2,:]
    
    # c) Grafica final(magnitud) 
    cmap ="plasma"
    title = data2['file_name']
    title = title[:-5]
    title_name='Imagen Final(RMA)\n'+title
    direction = 'Imagen Final_RMA_'+title
    
    vmin = np.amin(20*np.log10(abs(Sf_n)))+55 #dB
    vmax = np.amax(20*np.log10(abs(Sf_n)))#-20
    
    fig, ax = plt.subplots()
    im=ax.imshow(20*np.log10(abs(Sf_n)),cmap,origin='lower',aspect='equal', extent=[x[0]-dx/2, x[-1]+dx/2, y1-dy/2, y2+dy/2], vmin=vmin, vmax=vmax)
    ax.set(xlabel='Azimut(m)',ylabel='Rango(m)', title=title_name)
    ax.grid()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1) # pad es el espaciado con la grafica principal
    plt.colorbar(im,cax=cax,label='Reflectividad(dB)',extend='both')
    fig.savefig(os.getcwd()+"/Results/Imagenes_finales_reconstruidas_RMA/"+direction,orientation='landscape')
            
    """dF.plotImage(Sf_n,cmap=cmap,xlabel_name='Azimut(m)',ylabel_name='Rango(m)', title_name='Resultado Algoritmo Range Migration',
                 x_min=x[0]-dx/2, x_max=x[-1]+dx/2, y_min=y1-dy/2, y_max=y2+dy/2,unit_bar='(dB)',log=True,vmin=vmin,
                 vmax=vmax)"""
    
    return {'Sf_n':Sf_n, 'x_min':x[0]-dx/2, 'x_max':x[-1]+dx/2, 'y_min':y1-dy/2, 'y_max':y2+dy/2}

def main(dset_name):
    plt.close('all') # Cerrar todas las figuras previas
    
    # Declaracion de variables
    f = hp.File(os.getcwd()+"/Data_set/"+dset_name,'r') # os.getcwd() get the current path of this script
    global dset
    dset = f['sar_dataset']
    prm = sp.get_parameters2(dset)
    date = prm['date'].decode('utf-8') # string data
    
    global c,fc,BW,Nf,Ls,Np,Ro,theta,W,H,dx,dy,yi,dp2,Npx,Npy,show,Rmax
    
    c,fc,BW,Nf = prm['c'],prm['fc'],prm['BW'],prm['Nf']
    Ls,Np,Ro,theta = prm['Ls'],prm['Np'],prm['Ro'],prm['theta']
    W,H,dx,dy,yi = prm['w'],prm['h'],prm['dw'],prm['dh'],prm['hi'] # Dimensiones finales de la imagen
    Rmax,dp2 = prm['Rmax'],prm['dx']
    Npx,Npy = int(np.ceil(W/dx+1)),int(np.ceil(Rmax/dy)) # Numero de pixeles finales si hay decimacion
    show = False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

    # Obtencion del Raw Data
    start_time = timeit.default_timer()
    datos = get_SAR_data() # Obtiene el historico de fase 
    print("Tiempo de simulación: ",timeit.default_timer() - start_time,"s")

    # Procesamiento RMA
    start_time = timeit.default_timer()
    d_p = RMA_Algorithm(datos) # Implementa el algoritmo RMA
    print("Tiempo del procesamiento: ",timeit.default_timer() - start_time,"s")

    # Graficas
    #d_p = {'Sf':np.load("Data_post_procesRMA4.npy"),'x':np.load('Eje_x4.npy'),'y':np.load('Eje_y4.npy')}
    d_p['R_max'] = datos['R_max']
    d_p['file_name'] = dset_name
    IF = plot_image(d_p) # Grafica de la magnitud
    IF['date'] = date # Se anexa al diccionario
    
    return IF
    
if __name__ == '__main__':
  test = main("dset_60.hdf5")  