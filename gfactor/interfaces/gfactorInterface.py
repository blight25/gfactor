# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:39:16 2022

@author: Ben Lightfoot and John Noonan - separate file meant for running 
a GUI while calculating g-factors.
"""
from g_factor_universal import g_factor
import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyreadstat

class gf_interface:
    
    def __init__(self):
        version = 1.0
        
        #Slider Plot
        self.fig1,self.ax1 = plt.subplots()
        plt.tight_layout()
        self.fig1.show()
        self.gf = g_factor()
        
        #Tkinter GUI
        self.root = tk.Tk()
        self.root.title("G-factor calculator")
        self.root.grid_columnconfigure(0, weight = 1)
        self.root.grid_rowconfigure((0,1,2), weight = 1)
        
        #TopFrame
        self.top_frame = tk.Frame(master = self.root, highlightbackground = "grey", highlightthickness = 3, borderwidth = 3)
        self.top_frame.grid(row = 0, column = 0, padx = 20, pady = 20, ipady = 10, sticky = "nsew")
        self.top_frame.grid_columnconfigure((0,1), weight = 1)
        self.top_frame.grid_rowconfigure((0,1,2), weight = 1)
        
        #MidFrame
        self.mid_frame = tk.Frame(master = self.root, highlightbackground = "grey", highlightthickness = 3, borderwidth = 3)
        self.mid_frame.grid(row = 1, column = 0, padx = 20, pady = 20, ipady = 10, sticky = "nsew")
        self.mid_frame.grid_columnconfigure((0,1), weight = 1)
        self.mid_frame.grid_rowconfigure((0,1,2), weight = 1)
        
        #BottomFrame
        self.bottom_frame = tk.Frame(master = self.root, highlightbackground = "grey", highlightthickness = 3, borderwidth = 3)
        self.bottom_frame.grid(row = 2, column = 0, padx = 20, pady = 20, ipady = 10, sticky = "nsew")
        self.bottom_frame.grid_columnconfigure((0,1), weight = 1)
        self.bottom_frame.grid_rowconfigure((0,1), weight = 1)
        
        #Labels
        tk.Label(self.top_frame, text = "Pick a constants file to analyze:").grid(row=0,column=0)
        tk.Label(self.top_frame, text = "Pick a heliocentric distance (AU):").grid(row=1,column=0)
        tk.Label(self.top_frame, text = "Pick a heliocentric velocity (km/s):").grid(row=2,column=0)
        tk.Label(self.mid_frame, text = "If you would like to save your results, enter a name for your savefile:").grid(row=0,column=0)
        tk.Label(self.mid_frame, text = "Pick a file type for your savefile:").grid(row=1,column=0)
        tk.Label(self.mid_frame, text = "Click to create your savefile").grid(row = 2, column = 0)
        tk.Label(self.bottom_frame, text = "Click to print your results to the console").grid(row = 0,column = 0)
        tk.Label(self.bottom_frame, text = "Click if you would like a 3D plot (WARNING: this will take a while):").grid(row = 1, column = 0, pady = 20)

        #Entries
        self.save_file_input = tk.Entry(self.mid_frame, width = 49, borderwidth = 5)
        self.save_file_input.grid(row=0,column=1, padx=10, pady=20)
    
        #Scales
        self.hel_d_input = tk.Scale(self.top_frame, orient='horizontal', from_=0.05, to=25, width = 30, borderwidth = 5, length = 500, resolution = .05)
        self.hel_d_input.grid(row=1,column=1)
        self.hel_d_input.bind("<ButtonRelease-1>",self.update_plot)
        self.hel_v_input = tk.Scale(self.top_frame, orient = 'horizontal',from_=-100, to=100, width = 30, borderwidth = 5, length = 500, resolution = .5)
        self.hel_v_input.grid(row=2,column=1)
        self.hel_v_input.bind("<ButtonRelease-1>", self.update_plot)
        
        #Dropdown menu
        constants_file_options = ["hcos_constants.txt"]
        self.constants_file = tk.StringVar(value = "hcos_constants.txt")
        self.drop_constants_file = tk.OptionMenu(self.top_frame,self.constants_file, *constants_file_options)
        self.drop_constants_file.grid(row = 0, column = 1, pady = 20)
        self.drop_constants_file.config(width = 44, borderwidth = 3, pady = 7)
        save_file_options = ["excel","txt","csv",".sav"]
        self.file_type = tk.StringVar(value = 'txt')
        self.drop_save_file = tk.OptionMenu(self.mid_frame,self.file_type,*save_file_options)
        self.drop_save_file.config(width = 44, borderwidth=3, pady = 5)
        self.drop_save_file.grid(row=1,column=1)
    
        #Buttons
        tk.Button(self.mid_frame, text = "Save results", width=49, borderwidth = 3, command = self.save_results).grid(row=2,column=1, padx=20,pady=20)
        tk.Button(self.bottom_frame, text = "Retrieve dataframe", width=49, borderwidth=3, command = self.get_gfs).grid(row=0,column=1, padx=20,pady=20)
        tk.Button(self.bottom_frame, text = "3D plot", width=49, borderwidth=3, command = self.three_dimensional).grid(row = 1, column = 1)
        
    def update_plot(self, event):
        self.ax1.cla()
        self.ax1.set_yscale('log')
        self.ax1.set_ylim(1e-16,1e-11)
        self.ax1.set_xlabel('Wavelength (Angstroms)')
        self.ax1.set_ylabel('g-factor (phts s^-1)')
        self.ax1.set_title("g-factor plot")
        new_dataframe = self.gf.g_factor_calc(self.constants_file.get(), float(self.hel_d_input.get()),
                                              float(self.hel_v_input.get()))
        self.ax1.scatter(new_dataframe.iloc[:,1],new_dataframe.iloc[:,2])
        self.fig1.canvas.draw_idle()
        
    def three_dimensional(self):
        self.fig2 = plt.figure()
        self.ax2 = plt.axes(projection = '3d')
        val_list = []
        
        for i in range(-100,101):
            result = self.gf.g_factor_calc(self.constants_file.get(), float(self.hel_d_input.get()), i)
            result['Heliocentric Velocity'] = i
            val_list.append(result)
        
        df = pd.concat(val_list)
        x = df.iloc[:,1]
        y = df.iloc[:,3]
        z = df.iloc[:,2]
        self.ax2.cla()
        self.ax2.set_zlim(1e-16,1e-11)
        self.ax2.set_zscale('log')
        self.ax2.set_xlabel('Wavelength (Angstroms)')
        self.ax2.set_ylabel('Heliocentric Velocity (km/s)')
        self.ax2.set_zlabel('g-factor (phts s^-1)')
        self.ax2.set_title('3D g-factor plot')
        self.ax2.scatter(x,y,z)
        plt.show()
                
    def get_gfs(self):
        dataframe = self.gf.g_factor_calc(self.constants_file.get(), float(self.hel_d_input.get()),
                                          float(self.hel_v_input.get()))
        print(dataframe)
        
    def save_results(self):
        save_file_name = self.save_file_input.get()
        save_file_type = self.file_type.get()
        dataframe = self.gf.g_factor_calc(self.constants_file.get(), float(self.hel_d_input.get()),
                                          float(self.hel_v_input.get()))
        if save_file_type == "excel":
            dataframe.to_excel(save_file_name + ".xlsx")
        elif save_file_type == "txt":
            np.savetxt(save_file_name + ".txt", dataframe.values, header="ION ID   wavelength (Angstroms)  g-factor (phts s^-1)", delimiter = '            ', fmt='%s')
        elif save_file_type == "csv":
            dataframe.to_csv(save_file_name)
        elif save_file_type == ".sav":
            dataframe.columns =['ION_ID','wavelength','g_factor']
            pyreadstat.write_sav(dataframe, save_file_name + ".sav")
        print('Your results have been saved')
            
interface = gf_interface()
interface.root.mainloop()