from tkinter import *
import tkinter as tk
from model import *

master = Tk()
master.geometry('700x600')

bias = 0
epochs = 1
nu_hidden = 0
learning_rate = 0
layers = list()
selected_fun = 0


def GUI():
    # creat_select_label(epochs,learning_rate)
    def select(xtext_box, ytext_box):
        e = Entry(master)
        e.place(x=xtext_box, y=ytext_box, width=200, height=30)
        e.focus_set()

    # _____if select bias______
    def print_selection():
        global bias
        if (var1.get() == 1):
            bias = True
        elif (var1.get() == 0):
            bias = False

    def sel():
        global selected_fun
        selection = "You selected the option " + str(var.get())
        label.config(text=selection)
        if (int(var.get()) == 1):
            selected_fun = 1
        else:
            selected_fun = 2

        # print(selected_fun)

    def call_model():
        global learning_rate, epochs, layers, nu_hidden

        # get_nu_neuron_for_each_hidden
        layerstring = (layer.get())
        layers = layerstring.split()
        intlayers = []
        for i in range(len(layers)):
            intlayers.append(int(layers[i]))

        # ----input_user---------
        layers = intlayers
        nu_hidden = len(layers)
        learning_rate = float(learning_r.get())
        epochs = int(epoch_l.get())

        #call_fun_from_model_take_input
        input_user( nu_hidden,layers , learning_rate ,epochs , bias,selected_fun)

    # ----ladels layers,neuron-----
    l = tk.Label(master, text='Enter number of nerouns for each hidden layer ', font=100)
    l.pack(anchor=tk.W, padx=10, pady=8)
    layer = tk.StringVar()
    e = tk.Entry(master, textvariable=layer).place(x=350, y=10, width=200, height=30)

    # ----ladels learning rate-----
    l = tk.Label(master, text='Enter learning rate', font=100)
    l.pack(anchor=tk.W, padx=10, pady=20)
    learning_r = tk.StringVar()
    v = tk.Entry(master, textvariable=learning_r).place(x=350, y=60, width=200, height=30)

    # ----ladels epochs-----
    l = tk.Label(master, text='Enter number of epochs', font=100)
    l.pack(anchor=tk.W, padx=10, pady=15)
    epoch_l = tk.StringVar()
    e = tk.Entry(master, textvariable=epoch_l).place(x=350, y=125, width=200, height=30)

    # ----ladels bias -----
    l = tk.Label(master, text='Do you want to add bias?', font=50)
    l.pack(anchor=tk.W, padx=10, pady=10)
    # ____CheckBox_Bias____
    var1 = tk.IntVar()
    c1 = tk.Checkbutton(master, text='Bias', variable=var1, onvalue=1, offvalue=0, font=50,command=print_selection).place(x=350, y=180)
    label = 0

    # ----ladels activation function-----
    l = tk.Label(master, text='Choose the activation function ', font=100)
    l.pack(anchor=tk.W, padx=10, pady=15)

    # ____RadioButton____
    var = IntVar()
    R1 = Radiobutton(master, text="Sigmoid", variable=var, font=40, value=1, command=sel)
    R1.pack(anchor=tk.W, padx=10, pady=15)
    R2 = Radiobutton(master, text="Tanh", variable=var, font=40, value=2, command=sel)
    R2.pack(anchor=tk.W, padx=10, pady=15)
    label = Label(master)
    label.pack()

    # _____Button submit______
    b = Button(master, text="Submit", width=22, height=1, font="20", command=call_model)
    b.pack(anchor=tk.W, padx=240, pady=20)

    # run_gui
    master.mainloop()

