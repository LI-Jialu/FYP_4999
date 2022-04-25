from tkinter import *
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
# from main import test


def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb   

def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    return filename

# Epoch value 
def get_epoch_value():
    return '{: .2f}'.format(epoch_value.get())
def epoch_slider_changed(event):
    epoch_value_label.configure(text=get_epoch_value())

# Tmsp value 
def get_tmsp_value():
    return '{: .2f}'.format(tmsp_value.get())
def tmsp_slider_changed(event):
    tmsp_value_label.configure(text=get_tmsp_value())

# Bcsz value 
def get_bcsz_value():
    return '{: .2f}'.format(bcsz_value.get())
def bcsz_slider_changed(event):
    bcsz_value_label.configure(text=get_bcsz_value())

# Bcsz value 
def get_lr_value():
    return '{: .2f}'.format(lr_value.get())
def lr_slider_changed(event):
    lr_value_label.configure(text=get_lr_value())



def train():
    tk.messagebox.showinfo("Training finished.")
        
    answer = messagebox.askquestion(title = "Before model test",message = "Do you want to test this model on test dataset?")

    # answer = messagebox.askquestion(title = "Before backteting",message = "Do you want to backtest this model on test dataset?")



master = Tk()
# Set the size of the tkinter window
master.geometry("370x580")
master.resizable(False, False)
# Configure the color of the window
master.configure(bg=_from_rgb((255, 255, 255)))
style = ttk.Style(master)
style.theme_use('classic')
style.configure('Test.TLabel', background= 'white')
# slider values
epoch_value = tk.IntVar()
tmsp_value = tk.IntVar()
bcsz_value = tk.IntVar()
lr_value = tk.IntVar()
model_type = tk.StringVar() 

# Epoch slider
epoch_slider_label = ttk.Label(master,background='white',text='Epoch number')
epoch_slider_label.place(x=30, y = 30)
#  slider
epoch_slider = ttk.Scale(master,from_=0,to=100,orient='horizontal',command=epoch_slider_changed,variable=epoch_value)
epoch_slider.place(x=150, y=30)
# value label
epoch_value_label = ttk.Label(master,background='white',text=get_epoch_value())
epoch_value_label.place(x=300, y=30)

# Timestamp slider
tmsp_slider_label = ttk.Label(master, background='white',text='Timestamp number')
tmsp_slider_label.place(x=30, y = 100)
#  slider
tmsp_slider = ttk.Scale(master,from_=0,to=100,orient='horizontal', command=tmsp_slider_changed,variable=tmsp_value)
tmsp_slider.place(x=150, y=100)
# value label
tmsp_value_label = ttk.Label(master,background='white',text=get_tmsp_value())
tmsp_value_label.place(x=300, y=100)


# Batch_size slider
bcsz_slider_label = ttk.Label(master, background='white',text='Batch size')
bcsz_slider_label.place(x=30, y = 170)
#  slider
bcsz_slider = ttk.Scale(master,from_=0,to=1000,orient='horizontal', command=bcsz_slider_changed,variable=bcsz_value)
bcsz_slider.place(x=150, y=170)
# value label
bcsz_value_label = ttk.Label(master,background='white',text=get_bcsz_value())
bcsz_value_label.place(x=300, y=170)

# lr slider
lr_slider_label = ttk.Label(master, background='white',text='Learning rate')
lr_slider_label.place(x=30, y = 240)
#  slider
lr_slider = ttk.Scale(master,from_=0,to=1000,orient='horizontal', command=lr_slider_changed,variable=lr_value)
lr_slider.place(x=150, y=240)
# value label
lr_value_label = ttk.Label(master,background='white',text=get_lr_value())
lr_value_label.place(x=300, y=240)

slider_label = ttk.Label(master, background='white',text='Model type')
slider_label.place(x=30, y =310)
comboExample = ttk.Combobox(master, 
                            values=["LSTM", 
                                    "GRU",
                                    "LSTM & Attention",
                                    "GRU & Attention"]) 
comboExample.place(x=150, y =310)
comboExample.current(0)
model_type.set(comboExample.get())


Label(master, bg="white", text='Upload training file').place(x=30, y = 380)
tk.Button(master, text='Open', command=UploadAction).place(x=150, y = 380)

Label(master, bg="white", text='Upload testing file').place(x=30, y = 450)
tk.Button(master, text='Open', command=UploadAction).place(x=150, y = 450)

tk.Button(master, text='Start', command = train()).place(x=180, y = 520)


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

