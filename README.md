# DSM
DSM python module for Delta-Sigma Modulators modeling

    This module provides general classes for the high-level modeling of 
    discrete-time Delta-Sigma Modulators.
    
    A short example is also provided to clarify the use of the incuded classes. 

    Created on Thu Apr 27 15:55:26 2017

    @author: Michele Dei (michele.dei@imb-cnm.csic.es)

    tested on:    Python 2.7.12 (default, Nov 19 2016, 06:48:10) 
                  [GCC 5.4.0 20160609] on linux2

    17/05/08 JcisnerosFernandez:
        -3rd Order DSM Child including a Half-Delay FF description 
         as presented in http://hdl.handle.net/2117/89800 
        -Typical Power Spectral Stimation function "psd()"
        -Input dependent gain curve extraction from .csv
