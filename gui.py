from tkinter import *
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk


def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb   

def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)




master = Tk()
# Set the size of the tkinter window
master.geometry("1300x370")
master.resizable(False, False)
# Configure the color of the window
master.configure(bg=_from_rgb((240, 240, 240)))
 
# slider current value
current_value = tk.DoubleVar()


def get_current_value():
    return '{: .2f}'.format(current_value.get())


def slider_changed(event):
    value_label.configure(text=get_current_value())


# label for the slider
slider_label = ttk.Label(master, text='Epoch number')
slider_label.place(x=15, y = 30)
#  slider
slider = ttk.Scale(master,from_=0,to=100,orient='horizontal', command=slider_changed,variable=current_value)
slider.place(x=120, y=30)
# value label
value_label = ttk.Label(master,text=get_current_value())
value_label.place(x=240, y=30)

# label for the slider
slider_label = ttk.Label(master, text='Timestamp number')
slider_label.place(x=15, y = 130)
#  slider
slider = ttk.Scale(master,from_=0,to=100,orient='horizontal', command=slider_changed,variable=current_value)
slider.place(x=120, y=130)
# value label
value_label = ttk.Label(master,text=get_current_value())
value_label.place(x=240, y=130)


Label(master, bg="white", text='Upload file').place(x=15, y = 170)
tk.Button(master, text='Open', command=UploadAction).place(x=130, y = 170)

path = './1800_timestamp.png'
#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img1 = ImageTk.PhotoImage(Image.open(path).resize((480,320), Image.ANTIALIAS))
#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
Label(master, bg="white", text='Image of the predicted value v.s. true value').place(x=300, y = 10)
tk.Label(master, bg="white", image = img1).place(x=300,y=30)

path = './180_timestamp.png'
#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
img2 = ImageTk.PhotoImage(Image.open(path).resize((480,320), Image.ANTIALIAS))
#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
Label(master, bg="white", text='Image of backtesting').place(x=800, y = 10)
tk.Label(master, bg="white", image = img2).place(x=800,y=30)


mainloop()