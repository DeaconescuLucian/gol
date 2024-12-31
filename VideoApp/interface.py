import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import ttk
import cv2
from process import animate
from threading import *
import numpy as np

purple = "#5132b8"
pink = "#7e5087"
border_color = "#15ff00"
FONT_FAMILY = "Roboto"
FONT_SIZE = 12
font_style = (FONT_FAMILY, FONT_SIZE)
image = None
coef = 0.01
destructor_coef = 0.025
global animationOn, generator, event
animationOn = False
generator = None
event = None

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
    global image
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))  # Adjust the size of the image as needed
        photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        label.config(image=photo)
        label.image = photo
        button.place_forget()
        x_button.place(relx=1.0, rely=0.0, anchor="ne")

def remove_image():
    label.config(image="")
    button.place(relx=0.5, rely=0.9, anchor="s")
    x_button.place_forget()

def show_page(page_name):
    if page_name == "Process Images":
        container.pack()
    else:
        container.pack_forget()

def anim():
    global animationOn, generator, event, image
    if animationOn == False:
        animationOn = True
        button.place_forget()
    else:
        animationOn = False
        event.set()
        return

    button.place_forget()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    generator = animate(image, coef, destructor_coef)
    event = Event()

    t1 = Thread(target=get_images, args=(event,))
    t1.start()


def get_images(event):
    global image
    if animationOn == True:
        for new_frame in generator:
            if event.is_set():
                generator.close()
                print(new_frame)
                new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(new_frame))
                label.config(image=photo)
                label.image = new_frame
                label.update_idletasks()
                break

            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(new_frame))
            label.config(image=photo)
            label.image = new_frame
            label.update_idletasks()
root = tk.Tk()
root.geometry("900x900")
root.config(bg=purple)

# Create menu
menu_frame = tk.Frame(root, bg=pink, height=3)
menu_frame.pack(side="top", fill="x")

# Create buttons in the menu
buttons = ["Video to Images", "Images to Video", "Process Images"]
for button_text in buttons:
    button = tk.Button(menu_frame, text=button_text, command=lambda btn=button_text: show_page(btn), bg=purple, fg="white", relief="flat", height=3, width=20, font=font_style)
    button.pack(side="left", padx=50)

# Create a special container
container = tk.Frame(root, width=512, height=512, bg=pink)
container.place(relx=0.5, rely=0.1, anchor="n")
container.pack_propagate(0)

# Create a label to display the image
label = tk.Label(container)
label.pack()

# Create a button to choose the image
button = tk.Button(container, text="Choose Image", command=open_image, fg="white", bg=purple, font=font_style,relief="flat")
button.place(relx=0.5, rely=0.5, anchor="s")

buttonAnim = tk.Button(root, text="Animate Image", command=anim, fg="white", bg=pink, font=font_style,relief="flat")
buttonAnim.place(relx=0.5, rely=0.75, anchor="s")

# Create an "X" button to remove the image
x_button = tk.Button(container, text="X", command=remove_image, fg="white", bg="#9107fa", font=font_style, bd=0, relief="flat")
x_button.place_forget()

# Create sliders
def update_coef(value):
    global coef
    coef = float(value)
    coef_entry.delete(0, tk.END)
    coef_entry.insert(0, str(coef)[:5])

def update_coef_entry(*args):
    value = coef_entry.get()
    if value:
        coef_slider.set(float(value))

def update_destructor_coef(value):
    global destructor_coef
    destructor_coef = float(value)
    destructor_coef_entry.delete(0, tk.END)
    destructor_coef_entry.insert(0, str(destructor_coef)[:5])

def update_destructor_coef_entry(*args):
    value = destructor_coef_entry.get()
    if value:
        destructor_coef_slider.set(float(value))

coef_frame = tk.Frame(root, bg=purple)
coef_frame.pack()

coef_label = tk.Label(coef_frame, text="Coef", bg=purple, fg="white", font=font_style)
coef_label.pack(side="left", padx=10)

coef_slider = ttk.Scale(coef_frame, from_=0, to=1, length=300, orient="horizontal", command=update_coef, style="Pink.Horizontal.TScale")
coef_slider.set(coef)
coef_slider.pack(side="left", padx=10)
coef_frame.place(relx=0.5, rely=0.80, anchor="s")

coef_entry = tk.Entry(coef_frame, bg=pink, font=font_style, width=8)
coef_entry.insert(0, str(coef))
coef_entry.pack(side="left")
coef_entry.bind("<Return>", update_coef_entry)

destructor_coef_frame = tk.Frame(root, bg=purple)
destructor_coef_frame.pack()

destructor_coef_label = tk.Label(destructor_coef_frame, text="Destructor Coef", bg=purple, fg="white", font=font_style)
destructor_coef_label.pack(side="left", padx=10)

destructor_coef_slider = ttk.Scale(destructor_coef_frame, from_=0, to=1, length=300, orient="horizontal", command=update_destructor_coef, style="Pink.Horizontal.TScale")
destructor_coef_slider.set(destructor_coef)
destructor_coef_slider.pack(side="left", padx=10)
destructor_coef_frame.place(relx=0.46, rely=0.85, anchor="s")

destructor_coef_entry = tk.Entry(destructor_coef_frame, bg=pink, font=font_style, width=8)
destructor_coef_entry.insert(0, str(destructor_coef))
destructor_coef_entry.pack(side="left")
destructor_coef_entry.bind("<Return>", update_destructor_coef_entry)

# Define slider styles
style = ttk.Style()
style.configure("Pink.Horizontal.TScale", background="dark gray", troughcolor=purple, slidercolor=purple)

root.mainloop()
