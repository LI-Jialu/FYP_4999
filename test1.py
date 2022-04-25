from tkinter import *
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk


def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb   

testwin = Tk()
testwin.geometry("600x450")
testwin.resizable(False, False)
testwin.configure(bg=_from_rgb((255, 255, 255)))
 
path = './ff_300_GRU.png'
#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img1 = ImageTk.PhotoImage(Image.open(path).resize((480,320), Image.ANTIALIAS))
print(img1)
#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
Label(testwin, bg="white", text='Image of the predicted change v.s. true change').place(x=200, y = 10)
Label(testwin, bg="white", image = img1).place(x=80,y=30)
Label(testwin, bg="white", text='MSE = ').place(x=250, y = 370)
Label(testwin, bg="white", text='573.4').place(x=350, y = 370)
Label(testwin, bg="white", text='R squared = ').place(x=250, y = 400)
Label(testwin, bg="white", text='0.80').place(x=350, y = 400)
# Label(master, bg="white", text='Prediction accuracy').place(x=200, y = 10)

mainloop()