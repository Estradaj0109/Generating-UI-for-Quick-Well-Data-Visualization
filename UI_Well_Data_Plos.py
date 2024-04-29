import tkinter as tk
import numpy as np
from tkinter import ttk, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageGrab  # Importar la función grab de la biblioteca PIL
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors




class HistogramScatterBoxModule(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.canvas_list = []  # Lista para almacenar instancias de lienzo de gráficas
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="Módulo de Histogramas, Scatter Plots y Box Plots")
        self.label.pack()

        # Contenedor para los botones
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack()

        # Botones para cargar datos
        self.histogram_button = ttk.Button(self.buttons_frame, text="Cargar Datos Histograma", command=self.load_histogram_data)
        self.histogram_button.pack(side=tk.LEFT, padx=5)  # Colocar el botón a la izquierda con un espacio de 5 píxeles

        self.scatter_button = ttk.Button(self.buttons_frame, text="Cargar Datos Scatter Plot", command=self.load_scatter_data)
        self.scatter_button.pack(side=tk.LEFT, padx=5)  # Colocar el botón a la izquierda con un espacio de 5 píxeles

        self.boxplot_button = ttk.Button(self.buttons_frame, text="Cargar Datos Box Plot", command=self.load_boxplot_data)
        self.boxplot_button.pack(side=tk.LEFT, padx=5)  # Colocar el botón a la izquierda con un espacio de 5 píxeles

        # Espaciado entre los botones para centrarlos
        self.buttons_frame.pack_configure(pady=10)

        # Área para mostrar visualizaciones
        self.visualization_area = tk.Frame(self)
        self.visualization_area.pack(expand=True, fill="both")

    @staticmethod
    def ask_columns(prompt, columns):
        root = tk.Toplevel()
        root.title("Seleccione una columna")
        root.geometry("300x150")
        root.attributes('-topmost', True)  # Mantener la ventana en la parte superior

        label = ttk.Label(root, text=prompt)
        label.pack(pady=5)

        column_selected = tk.StringVar(root)
        column_selected.set(columns[0])

        def on_ok():
            root.destroy()

        ok_button = ttk.Button(root, text="Aceptar", command=root.destroy)
        ok_button.pack(pady=5)

        root.wait_window(root)
        return column_selected.get()

    def ask_column(self, prompt, columns):
        root = tk.Toplevel()
        root.title("Seleccione una columna")
        root.geometry("300x150")
        root.attributes('-topmost', True)  # Mantener la ventana en la parte superior

        label = ttk.Label(root, text=prompt)
        label.pack(pady=5)

        column_selected = tk.StringVar(root)
        column_selected.set(columns[0])

        label = ttk.Label(root, text="Eje X:")
        label.pack(pady=5)
        x_menu = ttk.OptionMenu(root, column_selected, *columns)
        x_menu.pack(pady=5)

        def on_ok():
            root.destroy()

        ok_button = ttk.Button(root, text="Aceptar", command=on_ok)
        ok_button.pack(pady=5)

        root.wait_window(root)
        return column_selected.get()

    def load_histogram_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            print("Selected file for HistogramScatterBoxModule - Histogram:", file_path)
            self.select_columns_and_generate_histogram(file_path)

    def load_scatter_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            print("Selected file for HistogramScatterBoxModule - Scatter Plot:", file_path)
            self.select_columns_and_generate_scatterplot(file_path)

    def load_boxplot_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            print("Selected file for HistogramScatterBoxModule - Box Plot:", file_path)
            self.select_columns_and_generate_boxplot(file_path)

    def select_columns_and_generate_histogram(self, file_path):
        df = pd.read_csv(file_path, delimiter=";")
        x_column = self.ask_column("Seleccione una columna para el histograma", df.columns)
        if x_column:
            self.generate_histogram(df, x_column)

    def select_columns_and_generate_scatterplot(self, file_path):
        df = pd.read_csv(file_path, delimiter=";")
        x_column, y_column = self.ask_columns("Seleccione las columnas para los ejes x e y", df.columns)
        if x_column and y_column:
            self.generate_scatterplot(df, x_column, y_column)

    def select_columns_and_generate_boxplot(self, file_path):
        df = pd.read_csv(file_path, delimiter=";")
        x_column, y_column = self.ask_columns("Seleccione las columnas para los ejes x e y", df.columns)
        if x_column and y_column:
            self.generate_boxplot(df, x_column, y_column)


    def ask_columns(self, prompt, columns):
        root = tk.Toplevel()
        root.title("Seleccione las columnas")
        root.geometry("300x200")
        root.attributes('-topmost', True)

        label = ttk.Label(root, text=prompt)
        label.pack(pady=5)

        x_column_var = tk.StringVar(root)
        y_column_var = tk.StringVar(root)

        x_column_var.set(columns[0])
        y_column_var.set(columns[0])

        x_label = ttk.Label(root, text="Eje X:")
        x_label.pack(pady=5)
        x_menu = ttk.OptionMenu(root, x_column_var, *columns)
        x_menu.pack(pady=5)

        y_label = ttk.Label(root, text="Eje Y:")
        y_label.pack(pady=5)
        y_menu = ttk.OptionMenu(root, y_column_var, *columns)
        y_menu.pack(pady=5)

        def on_ok():
            root.destroy()

        ok_button = ttk.Button(root, text="Aceptar", command=on_ok)
        ok_button.pack(pady=5)

        root.wait_window(root)
        return x_column_var.get(), y_column_var.get()


    def generate_histogram(self, df, x_column):
        # Crear la gráfica de histograma
        plt.figure(figsize=(3, 2))
        plt.hist(df[x_column], bins=10)
        plt.title("Histograma")
        plt.xlabel("Valores de " + x_column)
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=45)
        plt.tick_params(axis='both', which='major', labelsize=8)  # Cambiar el tamaño de los ticks de los ejes
        plt.tight_layout()

        # Crear un lienzo para embeber la gráfica
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.visualization_area)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Agregar el lienzo a la lista y verificar el límite de tres gráficas
        self.canvas_list.append(canvas)
        if len(self.canvas_list) > 3:
        # Eliminar el lienzo más antiguo y su contenedor del área de visualización
            self.canvas_list[0].get_tk_widget().pack_forget()
            self.canvas_list.pop(0)

    def generate_scatterplot(self, df, x_column, y_column):
        # Crear la gráfica de dispersión
        plt.figure(figsize=(3, 2))
        plt.scatter(df[x_column], df[y_column])
        plt.title("Scatter Plot")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=45)
        plt.tick_params(axis='both', which='major', labelsize=8)  # Cambiar el tamaño de los ticks de los ejes
        plt.tight_layout()

    # Crear un lienzo para embeber la gráfica
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.visualization_area)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Agregar el lienzo a la lista y verificar el límite de tres gráficas
        self.canvas_list.append(canvas)
        if len(self.canvas_list) > 3:
        # Eliminar el lienzo más antiguo y su contenedor del área de visualización
            self.canvas_list[0].get_tk_widget().pack_forget()
            self.canvas_list.pop(0)


    def generate_boxplot(self, df, x_column, y_column):
    # Crear la gráfica de Box Plot
        plt.figure(figsize=(3, 2))
        boxplot = plt.boxplot(df.groupby(x_column)[y_column].apply(list))
        plt.title("Box Plot")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=45)
        plt.tick_params(axis='both', which='major', labelsize=8)  # Cambiar el tamaño de los ticks de los ejes
        plt.tight_layout()

    # Crear un lienzo para embeber la gráfica
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.visualization_area)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Agregar el lienzo a la lista y verificar el límite de tres gráficas
        self.canvas_list.append(canvas)
        if len(self.canvas_list) > 3:
        # Eliminar el lienzo más antiguo y su contenedor del área de visualización
            self.canvas_list[0].get_tk_widget().pack_forget()
            self.canvas_list.pop(0)

class DataSelectionWindow(tk.Toplevel):
    def __init__(self, master, data_columns, callback):
        super().__init__(master)
        self.data_columns = data_columns
        self.callback = callback
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="Select data columns to plot:")
        self.label.pack(pady=5)

        self.checkbuttons = {}
        for column in self.data_columns:
            var = tk.BooleanVar(value=True)
            check_button = ttk.Checkbutton(self, text=column, variable=var)
            check_button.pack(anchor="w", padx=10, pady=2)
            self.checkbuttons[column] = var

        self.confirm_button = ttk.Button(self, text="Confirm", command=self.confirm_selection)
        self.confirm_button.pack(pady=5)

    def confirm_selection(self):
        selected_columns = [column for column, var in self.checkbuttons.items() if var.get()]
        self.callback(selected_columns)
        self.destroy()

class YAxisSelectionWindow(tk.Toplevel):
    def __init__(self, master, data_columns, callback):
        super().__init__(master)
        self.data_columns = data_columns
        self.callback = callback
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="Select primary and secondary y-axis values:")
        self.label.pack(pady=5)

        self.primary_y_label = ttk.Label(self, text="Primary Y-axis:")
        self.primary_y_label.pack(anchor="w", padx=10, pady=2)

        self.primary_y_var = tk.StringVar()
        self.primary_y_dropdown = ttk.Combobox(self, textvariable=self.primary_y_var, values=self.data_columns)
        self.primary_y_dropdown.pack(anchor="w", padx=10, pady=2)

        self.secondary_y_label = ttk.Label(self, text="Secondary Y-axis:")
        self.secondary_y_label.pack(anchor="w", padx=10, pady=2)

        self.secondary_y_var = tk.StringVar()
        self.secondary_y_dropdown = ttk.Combobox(self, textvariable=self.secondary_y_var, values=self.data_columns)
        self.secondary_y_dropdown.pack(anchor="w", padx=10, pady=2)

        self.confirm_button = ttk.Button(self, text="Confirm", command=self.confirm_selection)
        self.confirm_button.pack(pady=5)

    def confirm_selection(self):
        primary_y = self.primary_y_var.get()
        secondary_y = self.secondary_y_var.get()
        self.callback(primary_y, secondary_y)
        self.destroy()

class TemporalEventsTableModule(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.create_widgets()
        self.load_data = None  # Initialize load_data as None initially
        self.primary_y_column = None
        self.secondary_y_column = None
        self.selected_event = None
        self.canvas = None

    def create_widgets(self):
        self.label = ttk.Label(self, text="Temporal Events Table Module")
        self.label.pack()

        self.visualization_area = tk.Frame(self)
        self.visualization_area.pack(expand=True, fill="both")

        self.table_area = tk.Frame(self)
        self.table_area.pack(expand=True, fill="both")

        self.scatter_button = ttk.Button(self, text="Load Data and Generate Scatter Plot", command=self.load_data_and_select_columns)
        self.scatter_button.pack(pady=5)

    def load_data_and_select_columns(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            print("Selected file:", file_path)
            try:
                self.load_data = pd.read_csv(file_path, delimiter=";")
                self.show_data_selection_window()
            except Exception as e:
                print("Error loading data:", e)

    def show_data_selection_window(self):
        columns = self.load_data.columns[1:]  # Exclude the first column (DATE)
        selection_window = DataSelectionWindow(self.master, columns, self.show_y_axis_selection_window)

    def show_y_axis_selection_window(self, selected_columns):
        y_axis_selection_window = YAxisSelectionWindow(self.master, selected_columns, self.generate_scatterplot)

    def generate_scatterplot(self, primary_y_column, secondary_y_column):
        if not primary_y_column and not secondary_y_column:
            return  # Do nothing if no columns are selected

        plt.clf()  # Clear the previous figure
        plt.figure(figsize=(4, 2))  # Set maximum figure size

        # Configure X-axis
        x_column = "DATE"
        x_values = self.load_data[x_column]

        # Configure primary and secondary axes
        primary_ax = plt.gca()
        secondary_ax = primary_ax.twinx()

        # Plot selected series on primary and secondary axes
        lines = []
        for col in self.load_data.columns[1:]:
            y_values = self.load_data[col]
            if y_values.dtype in [np.float64, np.int64]:
                if col in [primary_y_column, secondary_y_column]:
                    if col == primary_y_column:
                        line, = primary_ax.plot(x_values, y_values, label=col)
                    else:
                        line, = secondary_ax.plot(x_values, y_values, label=col)
                    lines.append(line)
                else:
                    line, = primary_ax.plot(x_values, y_values, alpha=0.5, label=col)
                    lines.append(line)

        # Create legend with all plotted series
        labels = [line.get_label() for line in lines]
        primary_ax.legend(lines, labels)

        plt.title("Scatter plot")
        plt.xlabel(x_column)
        plt.ylabel(secondary_y_column)
        primary_ax.set_ylabel(primary_y_column)

        # Rotate x-axis tick labels
        primary_ax.tick_params(axis='x', labelrotation=45)

        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()

        # Draw vertical red line on click
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None  # Reset canvas
        self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.visualization_area)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onclick)

        # Display table
        self.show_table()

    def show_table(self):
        if hasattr(self, 'table') and self.table:
            self.table.destroy()

        self.table = ttk.Treeview(self.table_area)
        self.table["columns"] = tuple(self.load_data.columns[:2])
        self.table.heading("#0", text="Index")
        self.table.column("#0", width=50)
        for col in self.load_data.columns[:2]:
            self.table.heading(col, text=col)
            self.table.column(col, width=100)
        
        for i, (date, obs) in enumerate(zip(self.load_data["DATE"], self.load_data["OBS"])):
            self.table.insert("", tk.END, text=str(i+1), values=(date, obs))

        self.table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def onclick(self, event):
        if event.inaxes == plt.gca() and event.xdata is not None:
            x_value = event.xdata
            if self.selected_event:
                self.selected_event.remove()
            self.selected_event = plt.axvline(x=x_value, color='r')
            plt.draw()

class FileSelectionWindow(tk.Toplevel):
    def __init__(self, master, files, callback):
        super().__init__(master)
        self.files = files
        self.callback = callback
        self.selected_files = []
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="Select file(s) to plot:")
        self.label.pack(pady=5)

        self.checkbuttons = {}
        for file in self.files:
            var = tk.BooleanVar(value=False)
            check_button = ttk.Checkbutton(self, text=file, variable=var)
            check_button.pack(anchor="w", padx=10, pady=2)
            self.checkbuttons[file] = var

        self.confirm_button = ttk.Button(self, text="Confirm", command=self.confirm_selection)
        self.confirm_button.pack(pady=5)

    def confirm_selection(self):
        self.selected_files = [file for file, var in self.checkbuttons.items() if var.get()]
        if len(self.selected_files) < 1:
            messagebox.showwarning("Warning", "Select at least one file.")
        else:
            self.callback(self.selected_files)
            self.destroy()


class XAxisSelectionWindow(tk.Toplevel):
    def __init__(self, master, data_columns, callback):
        super().__init__(master)
        self.data_columns = data_columns
        self.callback = callback
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="Select up to three X-axis values:")
        self.label.pack(pady=5)

        self.x_axis_vars = [tk.StringVar() for _ in range(3)]
        self.x_axis_dropdowns = [ttk.Combobox(self, textvariable=self.x_axis_vars[i], values=self.data_columns) for i in range(3)]
        for dropdown in self.x_axis_dropdowns:
            dropdown.pack(anchor="w", padx=10, pady=2)

        self.confirm_button = ttk.Button(self, text="Confirm", command=self.confirm_selection)
        self.confirm_button.pack(pady=5)

    def confirm_selection(self):
        selected_x_axes = [var.get() for var in self.x_axis_vars if var.get()]
        if len(selected_x_axes) > 3:
            messagebox.showwarning("Warning", "Select up to three X-axis values.")
        elif len(selected_x_axes) == 0:
            messagebox.showwarning("Warning", "Select at least one X-axis value.")
        else:
            self.callback(selected_x_axes)
            self.destroy()



class ColorPaletteSelectionWindow(tk.Toplevel):
    def __init__(self, master, num_palettes, callback):
        super().__init__(master)
        self.num_palettes = num_palettes
        self.callback = callback
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="Select color palettes:")
        self.label.pack(pady=5)

        self.palette_vars = [tk.StringVar(value="viridis") for _ in range(self.num_palettes)]
        self.palette_dropdowns = [ttk.Combobox(self, textvariable=self.palette_vars[i], values=cm.datad.keys()) for i in range(self.num_palettes)]
        for dropdown in self.palette_dropdowns:
            dropdown.pack(anchor="w", padx=10, pady=2)

        self.confirm_button = ttk.Button(self, text="Confirm", command=self.confirm_selection)
        self.confirm_button.pack(pady=5)

    def confirm_selection(self):
        selected_palettes = [var.get().replace("'", "").replace(",", "") for var in self.palette_vars]  # Eliminar comillas y comas adicionales
        self.callback(selected_palettes)
        self.destroy()


class WellDataVisualizationModule(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.create_widgets()
        self.loaded_data = []
        self.canvas = None
        self.plots = []

    def create_widgets(self):
        self.label = ttk.Label(self, text="Well Data Visualization Module")
        self.label.pack()

        self.load_button = ttk.Button(self, text="Load Data", command=self.load_data)
        self.load_button.pack(pady=5)

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack()

    def clear_plots(self):
        for plot in self.plots:
            plot.destroy()
        self.plots = []

    def load_data(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_paths:
            self.clear_plots()  # Clear all plots before loading new data
            self.selected_files = file_paths
            self.load_columns(file_paths)

    def load_columns(self, file_paths):
        self.loaded_data = []
        for file in file_paths:
            try:
                df = pd.read_csv(file, delimiter=";")
            except Exception as e:
                print("Error reading CSV file:", e)
                continue

            self.loaded_data.append(df)

        if self.loaded_data:
            self.show_data_selection_window()

    def show_data_selection_window(self):
        column_names = self.loaded_data[0].columns[1:].tolist()  # Obtener los nombres de las columnas
        self.XAxisSelectionWindow = XAxisSelectionWindow(self.master, column_names, self.show_palette_selection_window)

    def show_palette_selection_window(self, selected_x_axes):
        self.ColorPaletteSelectionWindow = ColorPaletteSelectionWindow(self.master, len(selected_x_axes), lambda palettes: self.plot_data(selected_x_axes, palettes))

    def plot_data(self, selected_x_axes, selected_palettes):
        print("Selected Color Palettes:", selected_palettes)

        num_plots = len(selected_palettes)
        fig, axs = plt.subplots(1, num_plots, figsize=(1*num_plots, 6), sharey=True)
        fig.subplots_adjust(wspace=0.5)  # Ajustar la distancia horizontal entre subplots

        for i, ax in enumerate(axs):
            min_val = np.inf
            max_val = -np.inf
            min_y = np.inf  # Variable para almacenar el valor mínimo de y

            for df in self.loaded_data:
                if selected_x_axes[i] in df.columns:
                    min_val = min(min_val, df[selected_x_axes[i]].min())
                    max_val = max(max_val, df[selected_x_axes[i]].max())
                    x_values = df[selected_x_axes[i]]
                    y_values = df.iloc[:, 0]
                    min_y = min(min_y, y_values.min())  # Actualizar el valor mínimo de y

            # Obtener el mínimo y el máximo de los valores de x
            x_min = 0
            x_max = max_val

            # Crear un rango continuo de valores entre 0 y el valor máximo en x
            x_range = np.linspace(x_min, x_max, 100)

            # Asignar un color a cada valor en el rango utilizando la paleta seleccionada
            colors = plt.cm.get_cmap(selected_palettes[i])(np.linspace(0, 1, len(x_range)))

            # Rellenar verticalmente con los colores asignados
            for j, color in enumerate(colors[:-1]):
                ax.fill_between(x_values, np.maximum(min_y, 0), x_range[j+1], where=(x_values <= 0), color=colors[j])  # Acotar el relleno entre la serie de datos y x=0

            ax.set_xlim(0, x_max)  # Ajustar los límites en el eje x
            ax.set_ylim(np.maximum(min_y, 0), None)  # Ajustar los límites en el eje y
            ax.set_xlabel(selected_x_axes[i])
            ax.set_ylabel("DEPTH")
            ax.grid(True)

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

   

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Visualizar Datos de Pozo")
        self.geometry("1000x600")  # Ajustar el tamaño de la ventana principal

        self.main_frame = tk.Frame(self)
        self.main_frame.pack(expand=True, fill="both")

        self.histogram_scatter_box_module = HistogramScatterBoxModule(self.main_frame)
        self.histogram_scatter_box_module.grid(row=0, column=0, sticky="nsew")

        self.temporal_events_table_module = TemporalEventsTableModule(self.main_frame)
        self.temporal_events_table_module.grid(row=1, column=0, sticky="nsew")

        self.well_data_visualization_module = WellDataVisualizationModule(self.main_frame)
        self.well_data_visualization_module.grid(row=0, column=1, rowspan=2, sticky="nsew")

        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        self.create_menu()

    def create_menu(self):
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Load Data for WellDataVisualizationModule", command=self.well_data_visualization_module.load_data)
        self.file_menu.add_command(label="Save Screenshot", command=self.save_screenshot)  # Nuevo comando para guardar la captura de pantalla

    def save_screenshot(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
        if file_path:
            try:
        # Capturar solo la ventana de la interfaz de usuario y guardarla como una imagen JPG en la ubicación seleccionada
                x = self.winfo_rootx() + self.main_frame.winfo_x()
                y = self.winfo_rooty() + self.main_frame.winfo_y()
                x1 = x + self.main_frame.winfo_width()
                y1 = y + self.main_frame.winfo_height()
                ImageGrab.grab(bbox=(x, y, x1, y1)).save(file_path)
                print("Screenshot saved successfully:", file_path)
            except Exception as e:
                print("Error saving screenshot:", e)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
