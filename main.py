import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import logic  # Импортируем твой logic.py
import cv2

class TomatoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomato & Egg Analyzer")
        self.root.geometry("1100x750")
        
        # Контейнеры для моделей и данных
        self.sam_generator = None
        self.rf_model = None
        self.reader = None
        
        self.current_objects = None
        self.img_shape = None
        self.image_path = None

        self.setup_ui()
        self.init_models()

    def init_models(self):
        """Загрузка тяжелых моделей при старте"""
        self.status_bar.config(text="Статус: Загрузка моделей (SAM, RF, OCR)...")
        self.root.update()
        try:
            # Вызываем функцию загрузки из logic.py
            self.sam_generator, self.rf_model, self.reader = logic.load_all_models()
            self.status_bar.config(text="Статус: Модели готовы к работе")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", f"Проверь папку weights и наличие библиотек!\n{e}")
            self.status_bar.config(text="Статус: Ошибка инициализации")

    def setup_ui(self):
        # Левая панель управления
        self.side_panel = tk.Frame(self.root, width=250, bg="#2c3e50")
        self.side_panel.pack(side="left", fill="y")

        tk.Label(self.side_panel, text="УПРАВЛЕНИЕ", fg="white", bg="#2c3e50", font=("Arial", 12, "bold")).pack(pady=20)

        # Кнопки со стилем ttk
        self.btn_load = ttk.Button(self.side_panel, text="Загрузить фото", command=self.load_image)
        self.btn_load.pack(fill="x", padx=10, pady=10)

        self.btn_analyze = ttk.Button(self.side_panel, text="Посчитать объекты", command=self.analyze)
        self.btn_analyze.pack(fill="x", padx=10, pady=10)

        self.btn_ocr = ttk.Button(self.side_panel, text="Распознать букву", command=self.do_ocr)
        self.btn_ocr.pack(fill="x", padx=10, pady=10)

        # Фрейм статистики
        self.stats_frame = tk.LabelFrame(self.side_panel, text=" Результаты ", fg="white", bg="#2c3e50", padx=10, pady=10)
        self.stats_frame.pack(fill="both", expand=True, padx=10, pady=20)
        
        self.res_label = tk.Label(self.stats_frame, text="Жду загрузки...", fg="#ecf0f1", bg="#2c3e50", justify="left", font=("Consolas", 10))
        self.res_label.pack()

        # Правая панель для вывода изображения
        self.display_panel = tk.Frame(self.root, bg="#34495e")
        self.display_panel.pack(side="right", expand=True, fill="both")
        
        self.img_display = tk.Label(self.display_panel, bg="#34495e")
        self.img_display.pack(expand=True)

        self.status_bar = tk.Label(self.root, text="Статус: Инициализация...", bd=1, relief="sunken", anchor="w")
        self.status_bar.pack(side="bottom", fill="x")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self.image_path = path
            self.current_objects = None # Сбрасываем старые результаты
            img = Image.open(path)
            img.thumbnail((800, 600))
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_display.config(image=self.tk_img)
            self.status_bar.config(text=f"Загружено: {path}")
            self.res_label.config(text="Фото готово к анализу")

    def analyze(self):
        if not self.image_path:
            messagebox.showwarning("Внимание", "Сначала выберите файл!")
            return
        if not self.sam_generator: return
        
        self.status_bar.config(text="Анализирую (SAM + RF)... Пожалуйста, подождите.")
        self.root.update()
        
        try:
            img_bgr = cv2.imread(self.image_path)
            img_orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # 1. Препроцессинг и сегментация
            data = logic.load_and_preprocess(self.image_path)
            raw_masks = logic.get_basic_masks(data['enhanced'], self.sam_generator)
            refined_masks = logic.refine_masks(img_orig_rgb, raw_masks)
            
            # 2. Классификация
            # Предполагаем, что в logic.py FEATURES — это список имен колонок
            results = logic.predict_and_process(img_orig_rgb, refined_masks, self.rf_model, logic.FEATURES)
            
            self.current_objects = results
            self.img_shape = img_orig_rgb.shape

            # --- ВИЗУАЛИЗАЦИЯ ДЛЯ ПРЕПОДАВАТЕЛЯ ---
            # --- ОКНО 1: СЕГМЕНТАЦИЯ ---
            seg_img = logic.draw_segmentation_only(img_orig_rgb, results)
            self.show_extra_window("Этап 1: Сегментация (SAM)", seg_img)

            # --- ОКНО 2: КЛАССИФИКАЦИЯ ---
            class_img = logic.draw_classification_only(img_orig_rgb, results)
            self.show_extra_window("Этап 2: Классификация (RF)", class_img)

            # 3. Подсчет
            counts = {"Red Tomato": 0, "Orange Tomato": 0, "Quail Egg": 0, "Shadow": 0}
            for obj in results:
                lbl = obj['label']
                if lbl in counts: counts[lbl] += 1
            
            text = f"Красные: {counts['Red Tomato']}\nОранжевые: {counts['Orange Tomato']}\nЯйца: {counts['Quail Egg']}"
            self.res_label.config(text=text)
            
            self.status_bar.config(text="Анализ завершен")
            
        except Exception as e:
            messagebox.showerror("Ошибка анализа", f"Что-то пошло не так:\n{e}")
            self.status_bar.config(text="Ошибка")

    def do_ocr(self):
        if not self.current_objects:
            messagebox.showwarning("Внимание", "Сначала нажмите 'Посчитать объекты'!")
            return
        
        self.status_bar.config(text="Восстановление скелета буквы и OCR...")
        self.root.update()
        
        try:
            # Вызываем твой «Final Boss»
            canvas, char = logic.reconstruct_letter_final_boss(self.img_shape, self.current_objects, self.reader)
            
            # Показываем результат OCR в новом окне или просто алертом
            messagebox.showinfo("Результат OCR", f"Система распознала букву: {char}")
            
            # Визуализируем "скелет", который скормили EasyOCR
            skeleton_img = Image.fromarray(canvas)
            skeleton_img.thumbnail((400, 400))
            self.skel_tk = ImageTk.PhotoImage(skeleton_img)
            
            # Показываем превью скелета в маленьком окне
            top = tk.Toplevel(self.root)
            top.title("Скелет буквы")
            tk.Label(top, image=self.skel_tk).pack()
            tk.Label(top, text=f"Символ: {char}", font=("Arial", 16, "bold")).pack()
            
            self.status_bar.config(text="Распознавание завершено")
        except Exception as e:
            messagebox.showerror("Ошибка OCR", f"Не удалось собрать букву:\n{e}")
            self.status_bar.config(text="Ошибка OCR")

    def show_extra_window(self, title, image_array):
        """Вспомогательная функция для создания новых окон с картинками"""
        win = tk.Toplevel(self.root)
        win.title(title)
        
        pil_img = Image.fromarray(image_array)
        pil_img.thumbnail((700, 500))
        tk_img = ImageTk.PhotoImage(pil_img)
        
        label = tk.Label(win, image=tk_img)
        label.image = tk_img # Сохраняем ссылку, чтобы garbage collector не удалил фото
        label.pack(padx=10, pady=10)
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = TomatoApp(root)
    root.mainloop()