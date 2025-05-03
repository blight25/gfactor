# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:21:17 2022

@author: Ben Lightfoot and John Noonan - another GUI for g-factor calculations,
but done with a custom version of tkinter with a modern aesthetic.
"""

from g_factor_universal import g_factor
import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat

class gf_interface(ctk.CTk):
    
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    
    WIDTH = 700
    HEIGHT = 520 
    
    
    
    def __init__(self):
        super().__init__()
        
        version = 1.0
        
        #Slider Plot
        self.fig1,self.ax1 = plt.subplots()
        plt.tight_layout()
        self.fig1.show()
        self.gf = g_factor()
        
        #GUI attributes
        self.title("G-factor calculator")
        self.geometry(f"{gf_interface.WIDTH}x{gf_interface.HEIGHT}")
        self.grid_columnconfigure(0, weight= 1)
        self.grid_rowconfigure(0, weight = 1)
        
        #TopFrame
        self.top_frame = ctk.CTkFrame(master = self, corner_radius = 10)
        self.top_frame.grid(row = 0, column = 0, padx = 20, pady = 20, sticky = "nswe")
        self.top_frame.grid_rowconfigure((0,1,2,3), weight = 1)
        self.top_frame.grid_columnconfigure((0,1,2,3), weight = 1)
        #BottomFrame
        self.bottom_frame = ctk.CTkFrame(master = self, corner_radius = 10)
        self.bottom_frame.grid(row = 1, column = 0, padx = 20, pady = 20, sticky = "nswe")
        self.bottom_frame.grid_rowconfigure((0,1,2), weight = 1)
        self.bottom_frame.grid_columnconfigure((0,1,2), weight = 1)
        
        #Labels
        self.constants_file_label = ctk.CTkLabel(master = self.top_frame, 
                                                 text = "Pick a constants file to analyze:", 
                                                 text_font = ("Roboto Medium", -14))
        self.constants_file_label.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'we')
        self.hel_d_label = ctk.CTkLabel(master = self.top_frame,
                                        text = "Pick a heliocentric distance (AU):",
                                        text_font = ("Roboto Medium", -14))
        self.hel_d_label.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'we')
        self.hel_v_label = ctk.CTkLabel(master = self.top_frame, 
                                        text = "Pick a heliocentric velocity (km/s):",
                                        text_font = ("Roboto Medium", -14))
        self.hel_v_label.grid(row = 2, column = 0, padx = 10, pady = 10, sticky = 'we')
        self.save_file_entry_label = ctk.CTkLabel(master = self.bottom_frame, 
                                                  text = "To save your results, enter a name for your savefile:",
                                                  text_font = ("Roboto Medium", -14))
        self.save_file_entry_label.grid(row = 0, column = 0, padx = 10, pady = 10, sticky = 'we')
        self.save_file_type_label = ctk.CTkLabel(master = self.bottom_frame,
                                                 text = "Pick a file type for your savefile:",
                                                 text_font = ("Roboto Medium", -14))
        self.save_file_type_label.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = 'we')
        
        
        #Entries
        self.save_file_entry = ctk.CTkEntry(master = self.bottom_frame,
                                                 width = 100,
                                                 text_font = ("Roboto Medium", -14))
        self.save_file_entry.grid(row = 0, column = 1, padx = 20, pady = 20, sticky = 'we')
        self.hel_d_variable = tk.StringVar()
        self.hel_d_variable.set('12.48')
        self.hel_d_tracker = ctk.CTkEntry(master = self.top_frame,
                                          width = 50,
                                          text_font = ("Roboto Medium", -14),
                                          textvariable = self.hel_d_variable)
        self.hel_d_tracker.grid(row = 1, column = 3, sticky = 'we', padx = 5)
        self.hel_v_variable = tk.StringVar()
        self.hel_v_variable.set('0')
        self.hel_v_tracker = ctk.CTkEntry(master = self.top_frame,
                                          width = 50,
                                          text_font = ("Roboto Medium", -14),
                                          textvariable = self.hel_v_variable)
        self.hel_v_tracker.grid(row = 2, column = 3, sticky = 'we', padx = 5)
        
        #Dropdown Menues
        constants_file_options = ["hcos_constants.txt"]
        self.constants_file = ctk.StringVar(value = "hcos_constants.txt")
        self.drop_constants_file = ctk.CTkOptionMenu(master = self.top_frame, values = constants_file_options, variable = self.constants_file, text_font = ("Roboto Medium", -13))
        self.drop_constants_file.grid(row = 0, column = 1, padx = 20, pady = 10, sticky = 'we')
        save_file_options = ["excel","txt","csv",".sav"]
        self.file_type = ctk.StringVar(value = "txt")
        self.drop_save_file = ctk.CTkOptionMenu(master = self.bottom_frame, values = save_file_options, variable = self.file_type, text_font = ("Roboto Medium", -14))
        self.drop_save_file.grid(row = 1, column = 1, padx = 20, pady = 10, sticky = 'we')
        
        #Sliders
        self.hel_d_slider = ctk.CTkSlider(master = self.top_frame, from_ = 0.05, to = 25, command = self.display_hel_d)
        self.hel_d_slider.grid(row = 1, column = 1, columnspan = 2, padx = 20, pady = 10, sticky = 'we')
        self.hel_d_slider.bind("<ButtonRelease-1>",self.update_plot)
        self.hel_v_slider = ctk.CTkSlider(master = self.top_frame, from_ = -100, to = 100, command = self.display_hel_v)
        self.hel_v_slider.grid(row = 2, column = 1, columnspan = 2, padx = 20, pady = 20, sticky = 'we')
        self.hel_v_slider.bind("<ButtonRelease-1>",self.update_plot)
        #Buttons
        self.dataframe_button = ctk.CTkButton(master = self.top_frame, text = "Retrieve Dataframe", text_font = ("Roboto Medium", -14), width = 250, command = self.get_gfs)
        self.dataframe_button.grid(row = 3, column = 0, padx = 20, pady = 30)
        self.three_dimensional_button = ctk.CTkButton(master = self.top_frame, text = "3D Plot (this will take a while)", text_font = ('Roboto Medium', -14), width = 250, command = self.three_dimensional)
        self.three_dimensional_button.grid(row = 3, column = 1, columnspan = 3, padx = 20, pady = 30)
        self.save_results_button = ctk.CTkButton(master = self.bottom_frame, text = "Save results", text_font = ("Roboto Medium", -14), width = 300, command = self.save_results)
        self.save_results_button.grid(row = 2, column = 0, columnspan = 3, padx = 20, pady = 30)
        
    def display_hel_d(self,val):
        distance = round(val, 2)
        self.hel_d_variable.set(str(distance))
        return
    
    def display_hel_v(self,val):
        velocity = round(val, 2)
        self.hel_v_variable.set(str(velocity))
        return
    
    def update_plot(self, event):
        self.ax1.cla()
        self.ax1.set_yscale('log')
        self.ax1.set_ylim(1e-16,1e-11)
        self.ax1.set_xlabel('Wavelength (Angstroms)')
        self.ax1.set_ylabel('g-factor (phts s^-1)')
        self.ax1.set_title("g-factor plot")
        new_dataframe = self.gf.g_factor_calc(self.constants_file.get(), float(self.hel_d_variable.get()),
                                              float(self.hel_v_variable.get()))
        self.ax1.scatter(new_dataframe.iloc[:,1],new_dataframe.iloc[:,2])
        self.fig1.canvas.draw_idle()
        
    def three_dimensional(self):
        self.fig2 = plt.figure()
        self.ax2 = plt.axes(projection = '3d')
        val_list = []
        
        for i in range(-100,101):
            result = self.gf.g_factor_calc(self.constants_file.get(), float(self.hel_d_variable.get()), i)
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
        dataframe = self.gf.g_factor_calc(self.constants_file.get(), float(self.hel_d_variable.get()),
                                          float(self.hel_v_variable.get()))
        print(dataframe)
        
    def save_results(self):
        save_file_name = self.save_file_entry.get()
        save_file_type = self.file_type.get()
        dataframe = self.gf.g_factor_calc(self.constants_file.get(), float(self.hel_d_variable.get()),
                                          float(self.hel_v_variable.get()))
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
interface.mainloop()
    
    
    
    
    
    
    
    
    
    
    
        
       