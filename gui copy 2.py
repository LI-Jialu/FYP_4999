from tkinter import *
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk


def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb   

master = Tk()
master.geometry("1400x650")
master.resizable(False, False)
master.configure(bg=_from_rgb((255, 255, 255)))
 
path_1 = './backtest_good.png'
path_2 = './return_good.png'
#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img1 = ImageTk.PhotoImage(Image.open(path_1).resize((960,640), Image.ANTIALIAS))
img2 = ImageTk.PhotoImage(Image.open(path_2).resize((320,200), Image.ANTIALIAS))
#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
Label(master, bg="white", text='Image of the backtesting result').place(x=600, y = 10)
Label(master, bg="white", image = img1).place(x=80,y=30)
Label(master, bg="white", image = img2).place(x=1000,y=80)
Label(master, bg="white", text='Return = ').place(x=1100, y = 350)
Label(master, bg="white", text='11.71%').place(x=1200, y = 350)
Label(master, bg="white", text='Max drawdown = ').place(x=1100, y = 400)
Label(master, bg="white", text='4.5%').place(x=1200, y = 400)
Label(master, bg="white", text='Trade accuracy = ').place(x=1100, y = 450)
Label(master, bg="white", text='65%').place(x=1200, y = 450)
Label(master, bg="white", text='Buy accuracy = ').place(x=1100, y = 500)
Label(master, bg="white", text='67%').place(x=1200, y = 500)
Label(master, bg="white", text='Sell accuracy = ').place(x=1100, y = 550)
Label(master, bg="white", text='62%').place(x=1200, y = 550)
# Label(master, bg="white", text='Prediction accuracy').place(x=200, y = 10)

'''path = './180_timestamp.png'
#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img2 = ImageTk.PhotoImage(Image.open(path).resize((480,320), Image.ANTIALIAS))
#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
Label(master, bg="white", text='Image of backtesting').place(x=800, y = 10)
tk.Label(master, bg="white", image = img2).place(x=800,y=30)'''


mainloop()