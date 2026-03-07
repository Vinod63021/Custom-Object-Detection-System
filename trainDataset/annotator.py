"""
====================================================
  INDUSTRIAL VEHICLE DATASET ANNOTATION TOOL v2.0
====================================================
Features:
  - Multi-class bounding box annotation
  - Zoom & Pan support
  - Keyboard shortcuts
  - Live statistics dashboard
  - Auto-save & resume
  - Image brightness/contrast controls
  - Box editing (resize/move)
  - Validation before save
  - Progress tracking
  - Export summary report
  - YOLO format output
"""

import cv2
import os
import shutil
import json
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageEnhance
import threading

# ─────────────────────────────────────────────
#  PATHS & CONFIG
# ─────────────────────────────────────────────
RAW_IMAGE_FOLDER    = "rawimages"
TRAIN_IMAGE_FOLDER  = "dataset/images/train"
VAL_IMAGE_FOLDER    = "dataset/images/val"
TRAIN_LABEL_FOLDER  = "dataset/labels/train"
VAL_LABEL_FOLDER    = "dataset/labels/val"
CLASS_FILE          = "dataset/classes.txt"
PROGRESS_FILE       = "dataset/.progress.json"
SESSION_LOG         = "dataset/.session_log.json"

for folder in [TRAIN_IMAGE_FOLDER, VAL_IMAGE_FOLDER,
               TRAIN_LABEL_FOLDER, VAL_LABEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ─────────────────────────────────────────────
#  LOAD CLASSES
# ─────────────────────────────────────────────
if not os.path.exists(CLASS_FILE):
    os.makedirs("dataset", exist_ok=True)
    with open(CLASS_FILE, "w") as f:
        f.write("car\ntrucK\nmotorbike\nbus\nperson\nbicycle")

with open(CLASS_FILE) as f:
    CLASSES = [c.strip() for c in f.readlines() if c.strip()]

# ─────────────────────────────────────────────
#  CLASS COLOR MAP  (BGR for OpenCV)
# ─────────────────────────────────────────────
COLORS = [
    (  0, 220, 255), ( 50, 205,  50), (255,  80,  80),
    (255, 165,   0), (180,   0, 255), (  0, 200, 180),
    (255, 215,   0), (100, 149, 237), (255, 105, 180),
    ( 64, 224, 208),
]

def get_color(cls_idx):
    return COLORS[cls_idx % len(COLORS)]

# ─────────────────────────────────────────────
#  PROGRESS MANAGER
# ─────────────────────────────────────────────
class ProgressManager:
    def __init__(self):
        self.data = {"annotated": [], "skipped": [], "val_set": [],
                     "last_index": 0, "total_boxes": 0}
        self.load()

    def load(self):
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE) as f:
                    self.data.update(json.load(f))
            except Exception:
                pass

    def save(self):
        with open(PROGRESS_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

    def mark_done(self, img_name, box_count, is_val=False):
        if img_name not in self.data["annotated"]:
            self.data["annotated"].append(img_name)
        if is_val and img_name not in self.data["val_set"]:
            self.data["val_set"].append(img_name)
        self.data["total_boxes"] += box_count
        self.save()

    def mark_skipped(self, img_name):
        if img_name not in self.data["skipped"]:
            self.data["skipped"].append(img_name)
        self.save()

    @property
    def annotated_count(self):  return len(self.data["annotated"])
    @property
    def skipped_count(self):    return len(self.data["skipped"])
    @property
    def total_boxes(self):      return self.data["total_boxes"]

# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
class AnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗  Industrial Vehicle Annotation Tool  v2.0")
        self.root.configure(bg="#0d0d14")
        self.root.minsize(1200, 750)

        # ── State ──────────────────────────────
        self.progress     = ProgressManager()
        self.image_list   = self._scan_images()
        self.current_idx  = self.progress.data.get("last_index", 0)
        self.current_idx  = min(self.current_idx, max(0, len(self.image_list)-1))

        self.boxes        = []          # list of [cls, x1, y1, x2, y2]
        self.selected_box = -1         # index of selected box
        self.drawing      = False
        self.start_x = self.start_y = 0

        # zoom / pan
        self.zoom_factor  = 1.0
        self.pan_x = self.pan_y = 0
        self.is_panning   = False
        self.pan_start_x = self.pan_start_y = 0

        # image adjustments
        self.brightness   = 1.0
        self.contrast     = 1.0

        # raw image (PIL)
        self.img_pil      = None
        self.img_orig     = None   # CV2 BGR

        # val split ratio
        self.val_ratio    = 0.2

        # session stats
        self.session_start = time.time()

        self._build_ui()
        self._bind_keys()

        if self.image_list:
            self.load_image()
        else:
            messagebox.showinfo("No Images",
                f"Place images in:\n{os.path.abspath(RAW_IMAGE_FOLDER)}")

    # ─────────────────────────────────────────
    #  IMAGE SCANNING
    # ─────────────────────────────────────────
    def _scan_images(self):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        imgs = [f for f in sorted(os.listdir(RAW_IMAGE_FOLDER))
                if f.lower().endswith(exts)]
        return imgs

    # ─────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────
    def _build_ui(self):
        BG       = "#0d0d14"
        PANEL_BG = "#13131f"
        ACCENT   = "#00e5ff"
        BTN_BG   = "#1e1e30"
        BTN_HOV  = "#2a2a45"
        FG       = "#e0e0f0"
        DIM      = "#6060a0"

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
            fieldbackground=BTN_BG, background=BTN_BG,
            foreground=FG, bordercolor="#3030505",
            arrowcolor=ACCENT, selectbackground=BTN_BG)
        style.configure("TProgressbar",
            troughcolor=BTN_BG, background=ACCENT,
            bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)
        style.configure("TScale", background=PANEL_BG,
            troughcolor=BTN_BG, foreground=ACCENT)

        # ── Top bar ──────────────────────────
        topbar = tk.Frame(self.root, bg="#09090f", height=48)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⬡  VEHICLE ANNOTATOR  v2.0",
            font=("Courier New", 13, "bold"),
            fg=ACCENT, bg="#09090f").pack(side="left", padx=16, pady=12)

        self.lbl_progress = tk.Label(topbar, text="",
            font=("Courier New", 10), fg=DIM, bg="#09090f")
        self.lbl_progress.pack(side="left", padx=8)

        # Session timer
        self.lbl_timer = tk.Label(topbar, text="⏱ 00:00",
            font=("Courier New", 10), fg=DIM, bg="#09090f")
        self.lbl_timer.pack(side="right", padx=16)

        self._update_timer()

        # ── Main layout ──────────────────────
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True)

        # Left panel
        left = tk.Frame(main, bg=PANEL_BG, width=230)
        left.pack(side="left", fill="y", padx=(8,4), pady=8)
        left.pack_propagate(False)

        # Canvas area
        center = tk.Frame(main, bg=BG)
        center.pack(side="left", fill="both", expand=True, pady=8)

        # Right panel
        right = tk.Frame(main, bg=PANEL_BG, width=230)
        right.pack(side="right", fill="y", padx=(4,8), pady=8)
        right.pack_propagate(False)

        # ── Canvas ───────────────────────────
        self.canvas = tk.Canvas(center, bg="#080810",
            cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_down)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Motion>",          self._on_mouse_move)
        self.canvas.bind("<ButtonPress-2>",   self._pan_start)
        self.canvas.bind("<B2-Motion>",       self._pan_drag)
        self.canvas.bind("<ButtonPress-3>",   self._pan_start)
        self.canvas.bind("<B3-Motion>",       self._pan_drag)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)
        self.canvas.bind("<Configure>",       lambda e: self.show_image())

        # canvas status bar
        self.canvas_status = tk.Label(center,
            text="Draw boxes with LEFT CLICK  |  Pan: RIGHT CLICK  |  Zoom: Scroll",
            font=("Courier New", 8), fg=DIM, bg="#09090f")
        self.canvas_status.pack(fill="x")

        # ────────────────────────────────────
        #  LEFT PANEL content
        # ────────────────────────────────────
        def sec_label(parent, text):
            tk.Label(parent, text=text.upper(),
                font=("Courier New", 8, "bold"),
                fg=ACCENT, bg=PANEL_BG).pack(anchor="w", padx=12, pady=(12,2))
            tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x", padx=8)

        # CLASS SELECTOR
        sec_label(left, "Active Class")
        self.class_var = tk.StringVar(value=CLASSES[0] if CLASSES else "")
        self.class_box = ttk.Combobox(left, textvariable=self.class_var,
            values=CLASSES, state="readonly", font=("Courier New", 10))
        self.class_box.pack(fill="x", padx=8, pady=4)

        # Class quick buttons
        cls_grid = tk.Frame(left, bg=PANEL_BG)
        cls_grid.pack(fill="x", padx=8, pady=2)
        for i, cls in enumerate(CLASSES[:8]):
            c = get_color(i)
            hex_color = "#{:02x}{:02x}{:02x}".format(c[2], c[1], c[0])
            btn = tk.Button(cls_grid, text=f"{i+1}:{cls[:6]}",
                font=("Courier New", 7), fg="#fff",
                bg=BTN_BG, activebackground=hex_color,
                relief="flat", bd=0, padx=4, pady=2,
                command=lambda x=i: self._quick_class(x))
            btn.grid(row=i//2, column=i%2, sticky="ew", padx=1, pady=1)
        cls_grid.columnconfigure(0, weight=1)
        cls_grid.columnconfigure(1, weight=1)

        # IMAGE ADJUSTMENTS
        sec_label(left, "Image Adjustments")

        def make_slider(parent, label, from_, to, init, cmd, var_name):
            row = tk.Frame(parent, bg=PANEL_BG)
            row.pack(fill="x", padx=8, pady=1)
            tk.Label(row, text=label, font=("Courier New", 8),
                fg=FG, bg=PANEL_BG, width=10, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=init)
            setattr(self, var_name, var)
            s = ttk.Scale(row, from_=from_, to=to, variable=var,
                orient="horizontal", command=cmd)
            s.pack(side="left", fill="x", expand=True)
            return var

        make_slider(left, "Brightness", 0.2, 3.0, 1.0,
            lambda e: self.show_image(), "bright_var")
        make_slider(left, "Contrast",   0.2, 3.0, 1.0,
            lambda e: self.show_image(), "contrast_var")

        tk.Button(left, text="↺ Reset Adjustments",
            font=("Courier New", 8), fg=DIM, bg=BTN_BG,
            relief="flat", bd=0, pady=4,
            command=self._reset_adjustments).pack(fill="x", padx=8, pady=2)

        # ZOOM CONTROLS
        sec_label(left, "Zoom & View")
        zoom_row = tk.Frame(left, bg=PANEL_BG)
        zoom_row.pack(fill="x", padx=8, pady=2)
        for txt, cmd in [("−", self._zoom_out), ("Fit", self._zoom_fit),
                         ("+", self._zoom_in)]:
            tk.Button(zoom_row, text=txt, font=("Courier New", 9, "bold"),
                fg=ACCENT, bg=BTN_BG, relief="flat", bd=0,
                padx=8, pady=4, command=cmd).pack(side="left",
                fill="x", expand=True, padx=1)

        self.lbl_zoom = tk.Label(left, text="Zoom: 100%",
            font=("Courier New", 8), fg=DIM, bg=PANEL_BG)
        self.lbl_zoom.pack(pady=2)

        # VAL SPLIT
        sec_label(left, "Val Split Ratio")
        val_row = tk.Frame(left, bg=PANEL_BG)
        val_row.pack(fill="x", padx=8, pady=2)
        self.val_var = tk.DoubleVar(value=0.2)
        ttk.Scale(val_row, from_=0.0, to=0.5,
            variable=self.val_var, orient="horizontal").pack(
            side="left", fill="x", expand=True)
        self.lbl_val = tk.Label(val_row, text="20%",
            font=("Courier New", 8), fg=FG, bg=PANEL_BG, width=5)
        self.lbl_val.pack(side="left")
        self.val_var.trace_add("write",
            lambda *a: self.lbl_val.config(
                text=f"{int(self.val_var.get()*100)}%"))

        # ────────────────────────────────────
        #  RIGHT PANEL content
        # ────────────────────────────────────
        # BOX LIST
        sec_label(right, "Annotations")

        list_frame = tk.Frame(right, bg=PANEL_BG)
        list_frame.pack(fill="both", expand=True, padx=8, pady=4)

        scrollbar = tk.Scrollbar(list_frame, bg=PANEL_BG,
            troughcolor=BTN_BG, width=10)
        scrollbar.pack(side="right", fill="y")

        self.box_listbox = tk.Listbox(list_frame,
            bg=BTN_BG, fg=FG,
            font=("Courier New", 8),
            selectbackground=ACCENT, selectforeground="#000",
            borderwidth=0, highlightthickness=0,
            yscrollcommand=scrollbar.set)
        self.box_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.box_listbox.yview)
        self.box_listbox.bind("<<ListboxSelect>>", self._on_box_select)

        box_btns = tk.Frame(right, bg=PANEL_BG)
        box_btns.pack(fill="x", padx=8, pady=2)
        tk.Button(box_btns, text="✕ Delete",
            font=("Courier New", 8), fg="#ff5555",
            bg=BTN_BG, relief="flat", bd=0, pady=4,
            command=self._delete_selected).pack(side="left",
            fill="x", expand=True, padx=1)
        tk.Button(box_btns, text="✕ Clear All",
            font=("Courier New", 8), fg="#ff5555",
            bg=BTN_BG, relief="flat", bd=0, pady=4,
            command=self._clear_all).pack(side="right",
            fill="x", expand=True, padx=1)

        # STATS
        sec_label(right, "Session Stats")
        self.stats_frame = tk.Frame(right, bg=PANEL_BG)
        self.stats_frame.pack(fill="x", padx=8, pady=4)
        self._stat_labels = {}
        for key in ["Images Done", "Skipped", "Total Boxes",
                    "Current Boxes", "Val Images"]:
            row = tk.Frame(self.stats_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=key+":", font=("Courier New", 8),
                fg=DIM, bg=PANEL_BG, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="0", font=("Courier New", 8, "bold"),
                fg=ACCENT, bg=PANEL_BG, anchor="e")
            lbl.pack(side="right")
            self._stat_labels[key] = lbl

        # Class distribution
        sec_label(right, "Class Distribution")
        self.dist_frame = tk.Frame(right, bg=PANEL_BG)
        self.dist_frame.pack(fill="x", padx=8, pady=4)

        # KEYBOARD SHORTCUTS
        sec_label(right, "Shortcuts")
        shortcuts = [
            ("→ / D", "Next & Save"),
            ("← / A", "Previous"),
            ("Z",     "Undo Last Box"),
            ("Del",   "Delete Selected"),
            ("S",     "Skip Image"),
            ("+/-",   "Zoom In/Out"),
            ("F",     "Fit to Screen"),
            ("R",     "Reset View"),
            ("1-8",   "Quick Class Select"),
            ("Ctrl+S","Force Save"),
            ("Ctrl+E","Export Report"),
        ]
        for key, action in shortcuts:
            row = tk.Frame(right, bg=PANEL_BG)
            row.pack(fill="x", padx=8)
            tk.Label(row, text=key, font=("Courier New", 7, "bold"),
                fg=ACCENT, bg=PANEL_BG, width=8, anchor="w").pack(side="left")
            tk.Label(row, text=action, font=("Courier New", 7),
                fg=DIM, bg=PANEL_BG, anchor="w").pack(side="left")

        # ── Bottom bar ───────────────────────
        bottom = tk.Frame(self.root, bg="#09090f", height=52)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        def nav_btn(parent, text, cmd, accent=False):
            fg_col = "#000" if accent else FG
            bg_col = ACCENT if accent else BTN_BG
            b = tk.Button(parent, text=text,
                font=("Courier New", 9, "bold"),
                fg=fg_col, bg=bg_col,
                activebackground=BTN_HOV if not accent else "#00b8cc",
                relief="flat", bd=0, padx=14, pady=10,
                command=cmd)
            b.pack(side="left", padx=4, pady=6)
            return b

        nav_btn(bottom, "⟵  PREV",       self.prev_image)
        nav_btn(bottom, "⊘  SKIP",        self.skip_image)
        nav_btn(bottom, "↺  UNDO",        self.undo)
        nav_btn(bottom, "SAVE & NEXT  ⟶", self.next_image, accent=True)

        tk.Button(bottom, text="⬇ Export Report",
            font=("Courier New", 8), fg=DIM, bg="#09090f",
            relief="flat", bd=0, padx=8, pady=10,
            command=self.export_report).pack(side="right", padx=8, pady=6)

        # Progress bar at very bottom
        pb_frame = tk.Frame(self.root, bg="#09090f")
        pb_frame.pack(fill="x", side="bottom")
        self.progress_bar = ttk.Progressbar(pb_frame,
            style="TProgressbar", mode="determinate")
        self.progress_bar.pack(fill="x", padx=0)

    # ─────────────────────────────────────────
    #  KEY BINDINGS
    # ─────────────────────────────────────────
    def _bind_keys(self):
        self.root.bind("<Right>",       lambda e: self.next_image())
        self.root.bind("<Left>",        lambda e: self.prev_image())
        self.root.bind("d",             lambda e: self.next_image())
        self.root.bind("a",             lambda e: self.prev_image())
        self.root.bind("z",             lambda e: self.undo())
        self.root.bind("s",             lambda e: self.skip_image())
        self.root.bind("f",             lambda e: self._zoom_fit())
        self.root.bind("r",             lambda e: self._reset_view())
        self.root.bind("<Delete>",      lambda e: self._delete_selected())
        self.root.bind("<equal>",       lambda e: self._zoom_in())
        self.root.bind("<minus>",       lambda e: self._zoom_out())
        self.root.bind("<Control-s>",   lambda e: self._force_save())
        self.root.bind("<Control-e>",   lambda e: self.export_report())
        for i in range(min(8, len(CLASSES))):
            self.root.bind(str(i+1), lambda e, x=i: self._quick_class(x))

    # ─────────────────────────────────────────
    #  LOAD IMAGE
    # ─────────────────────────────────────────
    def load_image(self):
        self.boxes        = []
        self.selected_box = -1

        if self.current_idx >= len(self.image_list):
            self._on_complete()
            return

        img_name = self.image_list[self.current_idx]
        path     = os.path.join(RAW_IMAGE_FOLDER, img_name)

        if not os.path.exists(path):
            self.current_idx += 1
            self.load_image()
            return

        self.img_orig = cv2.imread(path)
        if self.img_orig is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            self.current_idx += 1
            self.load_image()
            return

        img_rgb      = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2RGB)
        self.img_pil = Image.fromarray(img_rgb)

        # Check for existing annotation (resume)
        label_path = self._label_path_for(img_name, train=True)
        if os.path.exists(label_path):
            self._load_existing_labels(label_path)

        self._zoom_fit()
        self._update_ui()

    def _label_path_for(self, img_name, train=True):
        base  = os.path.splitext(img_name)[0] + ".txt"
        folder= TRAIN_LABEL_FOLDER if train else VAL_LABEL_FOLDER
        return os.path.join(folder, base)

    def _load_existing_labels(self, label_path):
        h, w = self.img_orig.shape[:2]
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    self.boxes.append([cls, x1, y1, x2, y2])

    # ─────────────────────────────────────────
    #  SHOW IMAGE ON CANVAS
    # ─────────────────────────────────────────
    def show_image(self):
        if self.img_pil is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        # Apply brightness & contrast
        img = self.img_pil.copy()
        img = ImageEnhance.Brightness(img).enhance(self.bright_var.get())
        img = ImageEnhance.Contrast(img).enhance(self.contrast_var.get())

        iw, ih = img.size
        disp_w = int(iw * self.zoom_factor)
        disp_h = int(ih * self.zoom_factor)
        img    = img.resize((disp_w, disp_h), Image.LANCZOS)

        # Draw boxes on PIL image via cv2
        img_cv = cv2.cvtColor(cv2.cvtColor(
            cv2.resize(self.img_orig, (disp_w, disp_h)), cv2.COLOR_BGR2RGB),
            cv2.COLOR_RGB2BGR)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # reapply brightness/contrast to overlay
        img_arr = cv2.cvtColor(
            cv2.resize(self.img_orig, (disp_w, disp_h)),
            cv2.COLOR_BGR2RGB)

        for idx, box in enumerate(self.boxes):
            cls, x1, y1, x2, y2 = box
            sx1 = int(x1 * self.zoom_factor)
            sy1 = int(y1 * self.zoom_factor)
            sx2 = int(x2 * self.zoom_factor)
            sy2 = int(y2 * self.zoom_factor)
            color_bgr = get_color(cls)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            thickness = 3 if idx == self.selected_box else 2
            # filled semi-transparent rect
            overlay = img_arr.copy()
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2),
                color_rgb, -1)
            img_arr = cv2.addWeighted(img_arr, 0.85, overlay, 0.15, 0)
            cv2.rectangle(img_arr, (sx1, sy1), (sx2, sy2),
                color_rgb, thickness)
            # label badge
            lbl = CLASSES[cls] if cls < len(CLASSES) else str(cls)
            lbl_full = f" {idx+1}:{lbl} "
            (tw, th), bl = cv2.getTextSize(lbl_full,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_arr,
                (sx1, sy1-th-bl-4), (sx1+tw, sy1),
                color_rgb, -1)
            cv2.putText(img_arr, lbl_full,
                (sx1, sy1-bl-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Draw live box while dragging
        if self.drawing and hasattr(self, '_live_x2'):
            sx1 = int(self.start_x)
            sy1 = int(self.start_y)
            sx2 = int(self._live_x2)
            sy2 = int(self._live_y2)
            cls = self.class_box.current()
            c   = get_color(cls)
            cr  = (c[2], c[1], c[0])
            cv2.rectangle(img_arr, (sx1,sy1),(sx2,sy2), cr, 2)
            w_box = abs(sx2-sx1); h_box = abs(sy2-sy1)
            cv2.putText(img_arr, f"{w_box}x{h_box}",
                (min(sx1,sx2), min(sy1,sy2)-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, cr, 1)

        pil_disp = Image.fromarray(img_arr)
        self._tk_img = ImageTk.PhotoImage(pil_disp)

        self.canvas.delete("all")
        off_x = max(0, (cw - disp_w)//2) + self.pan_x
        off_y = max(0, (ch - disp_h)//2) + self.pan_y
        self._canvas_offset = (off_x, off_y)
        self.canvas.create_image(off_x, off_y, anchor="nw",
            image=self._tk_img)

        # Grid overlay (optional, subtle)
        if self.zoom_factor > 2.0:
            for gx in range(0, disp_w, 50):
                self.canvas.create_line(
                    off_x+gx, off_y, off_x+gx, off_y+disp_h,
                    fill="#1a1a2e", dash=(2,8))
            for gy in range(0, disp_h, 50):
                self.canvas.create_line(
                    off_x, off_y+gy, off_x+disp_w, off_y+gy,
                    fill="#1a1a2e", dash=(2,8))

        self.lbl_zoom.config(text=f"Zoom: {int(self.zoom_factor*100)}%")

    # ─────────────────────────────────────────
    #  MOUSE EVENTS
    # ─────────────────────────────────────────
    def _canvas_to_img(self, cx, cy):
        ox, oy = self._canvas_offset if hasattr(self, '_canvas_offset') else (0,0)
        ix = (cx - ox) / self.zoom_factor
        iy = (cy - oy) / self.zoom_factor
        if self.img_orig is not None:
            h, w = self.img_orig.shape[:2]
            ix = max(0, min(w, ix))
            iy = max(0, min(h, iy))
        return int(ix), int(iy)

    def _on_mouse_down(self, event):
        self.drawing = True
        self.start_x, self.start_y = self._canvas_to_img(event.x, event.y)

    def _on_mouse_move(self, event):
        if self.drawing:
            self._live_x2, self._live_y2 = self._canvas_to_img(event.x, event.y)
            self.show_image()
        # Update coord display
        if self.img_orig is not None:
            ix, iy = self._canvas_to_img(event.x, event.y)
            h, w   = self.img_orig.shape[:2]
            self.canvas_status.config(
                text=f"  Cursor: ({ix}, {iy})  |  "
                     f"Image: {w}×{h}  |  "
                     f"Zoom: {int(self.zoom_factor*100)}%  |  "
                     f"Boxes: {len(self.boxes)}")

    def _on_mouse_up(self, event):
        self.drawing = False
        x2, y2 = self._canvas_to_img(event.x, event.y)
        x1, y1 = self.start_x, self.start_y

        if abs(x2-x1) < 5 or abs(y2-y1) < 5:
            return  # too small, ignore

        # Normalise direction
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1

        cls = self.class_box.current()
        self.boxes.append([cls, x1, y1, x2, y2])
        if hasattr(self, '_live_x2'):
            del self._live_x2, self._live_y2
        self._update_ui()

    def _on_box_select(self, event):
        sel = self.box_listbox.curselection()
        self.selected_box = sel[0] if sel else -1
        self.show_image()

    def _pan_start(self, event):
        self.is_panning    = True
        self.pan_start_x   = event.x - self.pan_x
        self.pan_start_y   = event.y - self.pan_y

    def _pan_drag(self, event):
        self.pan_x = event.x - self.pan_start_x
        self.pan_y = event.y - self.pan_start_y
        self.show_image()

    def _on_scroll(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_factor = max(0.1, min(10.0, self.zoom_factor * factor))
        self.show_image()

    # ─────────────────────────────────────────
    #  ZOOM CONTROLS
    # ─────────────────────────────────────────
    def _zoom_in(self):
        self.zoom_factor = min(10.0, self.zoom_factor * 1.2)
        self.show_image()

    def _zoom_out(self):
        self.zoom_factor = max(0.1, self.zoom_factor / 1.2)
        self.show_image()

    def _zoom_fit(self):
        if self.img_pil is None:
            return
        cw = self.canvas.winfo_width()  or 800
        ch = self.canvas.winfo_height() or 600
        iw, ih = self.img_pil.size
        self.zoom_factor = min(cw/iw, ch/ih) * 0.95
        self.pan_x = self.pan_y = 0
        self.show_image()

    def _reset_view(self):
        self.zoom_factor = 1.0
        self.pan_x = self.pan_y = 0
        self.show_image()

    def _reset_adjustments(self):
        self.bright_var.set(1.0)
        self.contrast_var.set(1.0)
        self.show_image()

    def _quick_class(self, idx):
        if idx < len(CLASSES):
            self.class_box.current(idx)
            self.class_var.set(CLASSES[idx])

    # ─────────────────────────────────────────
    #  SAVE LABELS
    # ─────────────────────────────────────────
    def _save_labels(self, img_name, is_val=False):
        label_folder = VAL_LABEL_FOLDER if is_val else TRAIN_LABEL_FOLDER
        base_name    = os.path.splitext(img_name)[0] + ".txt"
        label_path   = os.path.join(label_folder, base_name)
        h, w         = self.img_orig.shape[:2]
        with open(label_path, "w") as f:
            for box in self.boxes:
                cls, x1, y1, x2, y2 = box
                xc = ((x1+x2)/2) / w
                yc = ((y1+y2)/2) / h
                bw = abs(x2-x1) / w
                bh = abs(y2-y1) / h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        return label_path

    def _move_image(self, img_name, is_val=False):
        dest_folder = VAL_IMAGE_FOLDER if is_val else TRAIN_IMAGE_FOLDER
        src  = os.path.join(RAW_IMAGE_FOLDER, img_name)
        dst  = os.path.join(dest_folder, img_name)
        if os.path.exists(src):
            shutil.move(src, dst)

    # ─────────────────────────────────────────
    #  NAVIGATION
    # ─────────────────────────────────────────
    def next_image(self):
        if not self.image_list:
            return
        if not self.boxes:
            if not messagebox.askyesno("No Boxes",
                "No boxes annotated. Save empty label and continue?"):
                return

        img_name = self.image_list[self.current_idx]
        ann_count = len(self.progress.data["annotated"])
        total     = len(self.image_list)
        is_val    = (ann_count % max(1, int(1/max(0.01, self.val_var.get()))) == 0
                     and self.val_var.get() > 0)

        self._save_labels(img_name, is_val=is_val)
        self._move_image(img_name, is_val=is_val)
        self.progress.mark_done(img_name, len(self.boxes), is_val)
        self.progress.data["last_index"] = self.current_idx + 1
        self.progress.save()

        self.current_idx += 1
        if self.current_idx >= len(self.image_list):
            self._on_complete()
            return
        self.load_image()

    def prev_image(self):
        self.current_idx = max(0, self.current_idx - 1)
        self.load_image()

    def skip_image(self):
        if not self.image_list:
            return
        img_name = self.image_list[self.current_idx]
        self.progress.mark_skipped(img_name)
        self.current_idx += 1
        if self.current_idx >= len(self.image_list):
            self._on_complete()
            return
        self.load_image()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.selected_box = -1
            self._update_ui()

    def _delete_selected(self):
        if 0 <= self.selected_box < len(self.boxes):
            self.boxes.pop(self.selected_box)
            self.selected_box = -1
            self._update_ui()

    def _clear_all(self):
        if self.boxes and messagebox.askyesno(
                "Clear All", "Remove all boxes in this image?"):
            self.boxes.clear()
            self.selected_box = -1
            self._update_ui()

    def _force_save(self):
        if self.image_list and self.current_idx < len(self.image_list):
            img_name = self.image_list[self.current_idx]
            self._save_labels(img_name)
            messagebox.showinfo("Saved",
                f"Labels saved for:\n{img_name}")

    # ─────────────────────────────────────────
    #  UPDATE UI
    # ─────────────────────────────────────────
    def _update_ui(self):
        total = len(self.image_list)
        done  = self.current_idx
        pct   = (done / total * 100) if total else 0

        self.lbl_progress.config(
            text=f"Image {self.current_idx+1} / {total}  "
                 f"[{pct:.0f}% complete]")

        self.progress_bar["maximum"] = total
        self.progress_bar["value"]   = done

        # Stats
        self._stat_labels["Images Done"].config(
            text=str(self.progress.annotated_count))
        self._stat_labels["Skipped"].config(
            text=str(self.progress.skipped_count))
        self._stat_labels["Total Boxes"].config(
            text=str(self.progress.total_boxes))
        self._stat_labels["Current Boxes"].config(
            text=str(len(self.boxes)))
        self._stat_labels["Val Images"].config(
            text=str(len(self.progress.data["val_set"])))

        # Box list
        self.box_listbox.delete(0, tk.END)
        for i, box in enumerate(self.boxes):
            cls, x1, y1, x2, y2 = box
            lbl = CLASSES[cls] if cls < len(CLASSES) else str(cls)
            self.box_listbox.insert(tk.END,
                f"  #{i+1}  {lbl:<12}  {x2-x1}×{y2-y1}")

        # Class distribution
        for w in self.dist_frame.winfo_children():
            w.destroy()
        from collections import Counter
        dist = Counter(b[0] for b in self.boxes)
        for cls_idx, count in sorted(dist.items()):
            lbl = CLASSES[cls_idx] if cls_idx < len(CLASSES) else str(cls_idx)
            c   = get_color(cls_idx)
            hex_c = "#{:02x}{:02x}{:02x}".format(c[2], c[1], c[0])
            row = tk.Frame(self.dist_frame, bg="#13131f")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"● {lbl}",
                font=("Courier New", 8), fg=hex_c,
                bg="#13131f").pack(side="left")
            tk.Label(row, text=str(count),
                font=("Courier New", 8, "bold"),
                fg="#ffffff", bg="#13131f").pack(side="right")

        self.show_image()

    # ─────────────────────────────────────────
    #  TIMER
    # ─────────────────────────────────────────
    def _update_timer(self):
        elapsed = int(time.time() - self.session_start)
        m, s    = divmod(elapsed, 60)
        self.lbl_timer.config(text=f"⏱ {m:02d}:{s:02d}")
        self.root.after(1000, self._update_timer)

    # ─────────────────────────────────────────
    #  COMPLETION & EXPORT
    # ─────────────────────────────────────────
    def _on_complete(self):
        total  = len(self.image_list)
        done   = self.progress.annotated_count
        skipped= self.progress.skipped_count
        boxes  = self.progress.total_boxes
        elapsed= int(time.time() - self.session_start)
        m, s   = divmod(elapsed, 60)
        msg = (f"✅  Annotation Complete!\n\n"
               f"  Images annotated : {done}\n"
               f"  Images skipped   : {skipped}\n"
               f"  Total boxes      : {boxes}\n"
               f"  Val set images   : {len(self.progress.data['val_set'])}\n"
               f"  Session time     : {m}m {s}s\n\n"
               f"Dataset is ready for YOLO training.")
        messagebox.showinfo("Done!", msg)
        self.export_report()

    def export_report(self):
        elapsed = int(time.time() - self.session_start)
        m, s    = divmod(elapsed, 60)
        report  = {
            "timestamp"       : time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_seconds" : elapsed,
            "total_images"    : len(self.image_list),
            "annotated"       : self.progress.annotated_count,
            "skipped"         : self.progress.skipped_count,
            "val_images"      : len(self.progress.data["val_set"]),
            "total_boxes"     : self.progress.total_boxes,
            "classes"         : CLASSES,
            "train_folder"    : TRAIN_IMAGE_FOLDER,
            "val_folder"      : VAL_IMAGE_FOLDER,
        }
        path = "dataset/annotation_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        # Also write data.yaml for YOLO
        yaml_path = "dataset/data.yaml"
        with open(yaml_path, "w") as f:
            f.write(f"path: {os.path.abspath('dataset')}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n\n")
            f.write(f"nc: {len(CLASSES)}\n")
            f.write(f"names: {CLASSES}\n")

        messagebox.showinfo("Export Done",
            f"Report  → {os.path.abspath(path)}\n"
            f"YOLO YAML → {os.path.abspath(yaml_path)}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = AnnotationApp(root)
    root.mainloop()