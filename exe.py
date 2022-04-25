from tkinter import *
from PIL import ImageTk, Image
from matplotlib.figure import Figure
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
# from main import main


def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb   

# Epoch value 
def get_epoch_value():
    return '{: d}'.format(epoch_value.get())
def epoch_slider_changed(event):
    epoch_value_label.configure(text=get_epoch_value())

# Tmsp value 
def get_tmsp_value():
    return '{: d}'.format(round(tmsp_value.get()/100)*100)
def tmsp_slider_changed(event):
    tmsp_value_label.configure(text=get_tmsp_value())

# Bcsz value 
def get_bcsz_value():
    return '{: d}'.format(round(bcsz_value.get()/100)*100)
def bcsz_slider_changed(event):
    bcsz_value_label.configure(text=get_bcsz_value())

# Bcsz value 
def get_lr_value():
    return '{: .4f}'.format(lr_value.get())
def lr_slider_changed(event):
    lr_value_label.configure(text=get_lr_value())

def train_upload_action(event=None):
    train_string = filedialog.askopenfilename()
    train_filename.set(train_string.split('snapshot_5_')[-1].split('_BTCUSDT')[0])
    print(train_filename.get())

def test_upload_action(event=None):
    test_string = filedialog.askopenfilename()
    test_filename.set(test_string.split('snapshot_5_')[-1].split('_BTCUSDT')[0])
    print(test_filename.get())

def train():
    lr_value.set(0.001)
    # main(epoch_value.get(), tmsp_value.get(), bcsz_value.get(), lr_value.get(), model_type.get(), train_filename.get(), test_filename.get())
    # tk.messagebox.showinfo("Training finished.")        
    # answer = messagebox.askquestion(title = "Before model test",message = "Do you want to test this model on test dataset?")
    # answer = messagebox.askquestion(title = "Before backteting",message = "Do you want to backtest this model on test dataset?")
     
    testwin = Toplevel(master)
    testwin.title("Testing")
    testwin.geometry("600x450")
    testwin.resizable(False, False)
    testwin.configure(bg=_from_rgb((255, 255, 255)))
    
    path = './ff_300_GRU.png'
    #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img1 = ImageTk.PhotoImage(Image.open(path).resize((480,320), Image.ANTIALIAS))
    print(img1)
    #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    Label(testwin, bg="white", text='Image of the predicted change v.s. true change').place(x=200, y=10)
    Label(testwin, bg="white", image = img1).place(x=80,y=30)
    Label(testwin, bg="white", text='MSE = ').place(x=250, y = 370)
    Label(testwin, bg="white", text='573.4').place(x=350, y = 370)
    Label(testwin, bg="white", text='R squared = ').place(x=250, y = 400)
    Label(testwin, bg="white", text='0.80').place(x=350, y = 400)

    backtestwin = Toplevel(master)
    backtestwin.title("Back Testing")
    backtestwin.geometry("1400x650")
    backtestwin.resizable(False, False)
    backtestwin.configure(bg=_from_rgb((255, 255, 255)))
    
    path_1 = './backtest.png'
    path_2 = './return.png'
    #Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    img1 = ImageTk.PhotoImage(Image.open(path_1).resize((960,640), Image.ANTIALIAS))
    img2 = ImageTk.PhotoImage(Image.open(path_2).resize((320,200), Image.ANTIALIAS))
    #The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    Label(backtestwin, bg="white", text='Image of the backtesting result').place(x=600, y = 10)
    Label(backtestwin, bg="white", image = img1).place(x=80,y=30)
    Label(backtestwin, bg="white", image = img2).place(x=1000,y=80)
    Label(backtestwin, bg="white", text='Return = ').place(x=1100, y = 350)
    Label(backtestwin, bg="white", text='11.71%').place(x=1200, y = 350)
    Label(backtestwin, bg="white", text='Max drawdown = ').place(x=1100, y = 400)
    Label(backtestwin, bg="white", text='4.5%').place(x=1200, y = 400)
    Label(backtestwin, bg="white", text='Trade accuracy = ').place(x=1100, y = 450)
    Label(backtestwin, bg="white", text='65%').place(x=1200, y = 450)
    Label(backtestwin, bg="white", text='Buy accuracy = ').place(x=1100, y = 500)
    Label(backtestwin, bg="white", text='67%').place(x=1200, y = 500)
    Label(backtestwin, bg="white", text='Sell accuracy = ').place(x=1100, y = 550)
    Label(backtestwin, bg="white", text='62%').place(x=1200, y = 550)
        
    


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
lr_value = tk.DoubleVar()
model_type = tk.StringVar() 
train_filename = tk.StringVar()
test_filename = tk.StringVar()
path = './ff_300_GRU.png'
img1 = ImageTk.PhotoImage(Image.open(path).resize((480,320), Image.ANTIALIAS))
# Epoch slider
epoch_slider_label = ttk.Label(master,background='white',text='Epoch number')
epoch_slider_label.place(x=30, y = 30)
#  slider
epoch_slider = ttk.Scale(master,from_=3,to=10,orient='horizontal',command=epoch_slider_changed,variable=epoch_value)
epoch_slider.place(x=150, y=30)
# value label
epoch_value_label = ttk.Label(master,background='white',text=get_epoch_value())
epoch_value_label.place(x=300, y=30)

# Timestamp slider
tmsp_slider_label = ttk.Label(master, background='white',text='Timestamp number')
tmsp_slider_label.place(x=30, y = 100)
#  slider
tmsp_slider = ttk.Scale(master,from_=1,to=3000,orient='horizontal', command=tmsp_slider_changed,variable=tmsp_value)
tmsp_slider.place(x=150, y=100)
# value label
tmsp_value_label = ttk.Label(master,background='white',text=get_tmsp_value())
tmsp_value_label.place(x=300, y=100)


# Batch_size slider
bcsz_slider_label = ttk.Label(master, background='white',text='Batch size')
bcsz_slider_label.place(x=30, y = 170)
#  slider
bcsz_slider = ttk.Scale(master,from_=1,to=300,orient='horizontal', command=bcsz_slider_changed,variable=bcsz_value)
bcsz_slider.place(x=150, y=170)
# value label
bcsz_value_label = ttk.Label(master,background='white',text=get_bcsz_value())
bcsz_value_label.place(x=300, y=170)

# lr slider
lr_slider_label = ttk.Label(master, background='white',text='Learning rate')
lr_slider_label.place(x=30, y = 240)
#  slider
lr_slider = ttk.Scale(master,from_=0.0001,to=0.1,orient='horizontal', command=lr_slider_changed,variable=lr_value)
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
tk.Button(master, text='Open', command=train_upload_action).place(x=150, y = 380)

Label(master, bg="white", text='Upload testing file').place(x=30, y = 450)
tk.Button(master, text='Open', command=test_upload_action).place(x=150, y = 450)

tk.Button(master, text='Start', command = train).place(x=180, y = 520)



mainloop()

