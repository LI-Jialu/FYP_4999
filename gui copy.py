from tkinter import *
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb   

def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)

def get_current_value():
    return '{: .2f}'.format(current_value.get())


def slider_changed(event):
    value_label.configure(text=get_current_value())




master = Tk()
# Set the size of the tkinter window
master.geometry("370x580")
master.resizable(False, False)
# Configure the color of the window
master.configure(bg=_from_rgb((255, 255, 255)))
style = ttk.Style(master)
style.theme_use('classic')
style.configure('Test.TLabel', background= 'white')
# slider current value
current_value = tk.DoubleVar()

answer = messagebox.askquestion(title = "Before model test",message = "Do you want to test this model on test dataset?")

answer = messagebox.askquestion(title = "Before backteting",message = "Do you want to backtest this model on test dataset?")


# label for the slider
slider_label = ttk.Label(master, background='white', text='Epoch number')
slider_label.place(x=30, y = 30)
#  slider
slider = ttk.Scale(master,from_=0,to=100,orient='horizontal', command=slider_changed,variable=current_value)
slider.place(x=150, y=30)
# value label
value_label = ttk.Label(master,background='white',text=get_current_value())
value_label.place(x=300, y=30)

# label for the slider
slider_label = ttk.Label(master, background='white',text='Timestamp number')
slider_label.place(x=30, y = 100)
#  slider
slider = ttk.Scale(master,from_=0,to=100,orient='horizontal', command=slider_changed,variable=current_value)
slider.place(x=150, y=100)
# value label
value_label = ttk.Label(master,background='white',text=get_current_value())
value_label.place(x=300, y=100)


# label for the slider
slider_label = ttk.Label(master, background='white',text='Batch size')
slider_label.place(x=30, y = 170)
#  slider
slider = ttk.Scale(master,from_=0,to=1000,orient='horizontal', command=slider_changed,variable=current_value)
slider.place(x=150, y=170)
# value label
value_label = ttk.Label(master,background='white',text=get_current_value())
value_label.place(x=300, y=170)

# label for the slider
slider_label = ttk.Label(master, background='white',text='Learning rate')
slider_label.place(x=30, y = 240)
#  slider
slider = ttk.Scale(master,from_=0,to=1000,orient='horizontal', command=slider_changed,variable=current_value)
slider.place(x=150, y=240)
# value label
value_label = ttk.Label(master,background='white',text=get_current_value())
value_label.place(x=300, y=240)

slider_label = ttk.Label(master, background='white',text='Model type')
slider_label.place(x=30, y =310)
entry1 = tk.Entry (master).place(x=150, y=310)

Label(master, bg="white", text='Upload training file').place(x=30, y = 380)
tk.Button(master, text='Open', command=UploadAction).place(x=150, y = 380)

Label(master, bg="white", text='Upload testing file').place(x=30, y = 450)
tk.Button(master, text='Open', command=UploadAction).place(x=150, y = 450)

tk.Button(master, text='Start', command=UploadAction).place(x=180, y = 520)

'''
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
tk.Label(master, bg="white", image = img2).place(x=800,y=30)'''


mainloop()

