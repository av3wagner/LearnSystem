#https://csatlas.com/python-import-file-module/
import streamlit as st  
import streamlit.components.v1 as components
from  PIL import Image
import numpy as np
import pandas as pd
import base64
import sys
import inspect, os
import pathlib
from os import listdir
from os.path import isfile, join
import glob
import os
import subprocess
st.set_page_config(page_title="Clinica Reports")

def execute_python_file1(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            python_code = file.read()
            exec(python_code)
    except FileNotFoundError:
        st.markdown(f"Error: The file '{file_path}' does not exist.")


def execute_python_file(file_path):
    try:
        completed_process = subprocess.run(['python', file_path], capture_output=True, text=True)
        if completed_process.returncode == 0:
            st.markdown("Execution successful.")
            st.markdown("Output:")
            st.markdown(completed_process.stdout)
        else:
            st.markdown(f"Error: Failed to execute '{file_path}'.")
            st.markdown("Error output:")
            st.markdown(completed_process.stderr)
    except FileNotFoundError:
        st.markdown(f"Error: The file '{file_path}' does not exist.")
        
file_path ='C:\IPYNBgesamt2025\ASNI-FEN\ASNI-FEN-SYSTEM\subdir\EDAReports.py' 
execute_python_file1(file_path)

#st.markdown("Zweite Versucht")
#st.markdown('C:\IPYNBgesamt\ASNI-FEN\ASNI-FEN-SYSTEM\subdir\Hello.py')
#import importlib
#importlib.import_module('hello')
#subdir.mymodule.say_hello()

#from subdir.mymodule import say_hello
#say_hello()

#import subdir.mymodule as m
#m.say_hello()

