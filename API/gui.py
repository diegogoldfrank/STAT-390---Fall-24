import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from skimage import morphology
import gc
import threading
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Tool")
        self.root.geometry("900x1200")
        
        # Initialize match-related variables
        self.he_folder = None
        self.melan_folder = None
        self.sox10_folder = None
        
        # Create main frames
        self.setup_gui()
        
        # Processing queue for segmentation
        self.processing_queue = []
        self.currently_processing = False
        self.current_row = 0
        
        # Store current images for cleanup
        self.current_image_labels = []
        
        # Bind scroll events
        self.bind_scroll_events()
        
    def bind_scroll_events(self):
        # Windows and macOS trackpad
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows)
        # Linux
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)
        
    def _on_mousewheel_windows(self, event):
        delta = event.delta
        if abs(delta) >= 120:  # Windows
            delta = int(-delta/120)
        else:  # macOS
            delta = -delta
        self.canvas.yview_scroll(delta, "units")
        
    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
    
    def setup_styles(self):
        # Configure styles for different elements
        style = ttk.Style()
        style.configure('Large.TButton', font=('Arial', 14), padding=(10, 15))
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Regular.TLabel', font=('Arial', 14))
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'))  # Increased font size for titles

    def setup_gui(self):
        # Add styles setup
        self.setup_styles()
        
        # Control frame at top
        self.control_frame = ttk.Frame(self.root, padding="20")
        self.control_frame.pack(fill=tk.X)
        
        # Changed order: First Match frame, then Segmentation frame
        self.setup_match_frame()
        self.setup_segmentation_frame()
        
        # Progress label
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(
            self.control_frame,
            textvariable=self.progress_var,
            font=('Arial', 14)
        ).pack(pady=20)
        
        # Create scrollable frame for results
        self.setup_scrollable_frame()

    def setup_match_frame(self):
        # Frame for Match functionality
        self.match_frame = ttk.LabelFrame(
            self.control_frame,
            padding=(20, 20)
        )
        self.match_frame.pack(fill="x", padx=20, pady=15)
        
        # Title label with larger font
        ttk.Label(
            self.match_frame,
            text="Step 2: Slice Matching",
            style='Title.TLabel'
        ).pack(pady=(10, 15))
        
        # Description label
        ttk.Label(
            self.match_frame,
            text="Match images across different stains (H&E, Melan, Sox10)",
            font=('Arial', 14)
        ).pack(pady=(10, 15))
        
        # Create frames for folder selection
        folders_frame = ttk.Frame(self.match_frame)
        folders_frame.pack(fill="x", pady=10)
        
        # H&E folder selection
        he_frame = ttk.Frame(folders_frame)
        he_frame.pack(fill="x", pady=5)
        ttk.Label(he_frame, text="H&E Folder:", font=('Arial', 14)).pack(side="left", padx=10)
        self.he_path_var = tk.StringVar()
        ttk.Label(he_frame, textvariable=self.he_path_var, font=('Arial', 14), width=50).pack(side="left", padx=10)
        ttk.Button(he_frame, text="Browse", command=self.select_he_folder, style='Large.TButton').pack(side="left", padx=10)
        
        # Melan folder selection
        melan_frame = ttk.Frame(folders_frame)
        melan_frame.pack(fill="x", pady=5)
        ttk.Label(melan_frame, text="Melan Folder:", font=('Arial', 14)).pack(side="left", padx=10)
        self.melan_path_var = tk.StringVar()
        ttk.Label(melan_frame, textvariable=self.melan_path_var, font=('Arial', 14), width=50).pack(side="left", padx=10)
        ttk.Button(melan_frame, text="Browse", command=self.select_melan_folder, style='Large.TButton').pack(side="left", padx=10)
        
        # Sox10 folder selection
        sox10_frame = ttk.Frame(folders_frame)
        sox10_frame.pack(fill="x", pady=5)
        ttk.Label(sox10_frame, text="Sox10 Folder:", font=('Arial', 14)).pack(side="left", padx=10)
        self.sox10_path_var = tk.StringVar()
        ttk.Label(sox10_frame, textvariable=self.sox10_path_var, font=('Arial', 14), width=50).pack(side="left", padx=10)
        ttk.Button(sox10_frame, text="Browse", command=self.select_sox10_folder, style='Large.TButton').pack(side="left", padx=10)
        
        # Process button
        ttk.Button(
            self.match_frame,
            text="Process Matching",
            command=self.process_matching,
            style='Large.TButton'
        ).pack(pady=20)

    def setup_segmentation_frame(self):
        # Frame for Segmentation functionality
        self.segmentation_frame = ttk.LabelFrame(
            self.control_frame,
            padding=(20, 20)
        )
        self.segmentation_frame.pack(fill="x", padx=20, pady=15)
        
        # Title label with larger font
        ttk.Label(
            self.segmentation_frame,
            text="Step 3: Segmentation of Epithelium and Stroma",
            style='Title.TLabel'
        ).pack(pady=(10, 15))
        
        # Description label
        ttk.Label(
            self.segmentation_frame,
            text="Process TIFF images to segment stroma and epithelia",
            font=('Arial', 14)
        ).pack(pady=(10, 15))
        
        # Buttons frame with padding
        buttons_frame = ttk.Frame(self.segmentation_frame)
        buttons_frame.pack(fill="x", pady=10)
        
        # Select files button
        ttk.Button(
            buttons_frame,
            text="Select Files",
            command=lambda: self.select_files(),
            style='Large.TButton'
        ).pack(side="left", padx=20)
        
        # Select folder button
        ttk.Button(
            buttons_frame,
            text="Select Folder",
            command=lambda: self.select_folder(),
            style='Large.TButton'
        ).pack(side="left", padx=20)

    def process_segmentation(self, files=None, folder=None):
        """Process images for segmentation of epithelia and stroma"""
        try:
            # Clear previous results
            self.clear_previous_results()
            
            # Get list of files to process
            files_to_process = []
            if files:
                files_to_process = list(files)
            elif folder:
                # Get all .tif files from folder
                files_to_process = [os.path.join(folder, f) for f in os.listdir(folder) 
                                if f.lower().endswith(('.tif', '.tiff'))]
            
            if not files_to_process:
                messagebox.showerror("Error", "No .tif files selected for processing")
                return
                
            # Create thread for processing
            thread = threading.Thread(
                target=self.run_segmentation_process,
                args=(files_to_process,)
            )
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error starting segmentation: {str(e)}")
            self.progress_var.set("Ready")

    def run_segmentation_process(self, files_to_process):
        """Run the segmentation process in a separate thread"""
        try:
            for file_path in files_to_process:
                self.progress_var.set(f"Processing: {os.path.basename(file_path)}")
                
                # Load and process image
                results = self.segment_image(file_path)
                
                # Display results in GUI
                self.root.after(0, self.display_segmentation_results, results, file_path)
                
            self.progress_var.set("Ready")
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Error in segmentation: {str(e)}")
            self.progress_var.set("Ready")

    def segment_image(self, image_path):
        """Implement the segmentation algorithm for a single image"""
        # Load Image
        img_rgb = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        # Convert to YCrCb
        img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    
        # Lumma binning
        lumma_bins_n = 20
        divisor = (np.floor(255 / lumma_bins_n).astype(np.uint8))
        lumma_binned = (np.floor(img_ycrcb[:,:,0]/divisor)).astype(np.uint8)

        # Find bin with most pixels (background)
        most_pixels_bin = -1
        most_pixels = 0
        for bin_i in range(0, lumma_bins_n+1):
            n_pixels = np.count_nonzero(lumma_binned == bin_i)
            if n_pixels > most_pixels:
                most_pixels = n_pixels
                most_pixels_bin = bin_i
        
        # Find background
        background_bin = most_pixels_bin
        background = lumma_binned == background_bin
        background = morphology.remove_small_objects(background, 5000)
        background = morphology.remove_small_holes(background, 10000)
        
        # Red Chroma binning
        Cr_bins_n = 50
        divisor = (np.floor(255 / Cr_bins_n).astype(np.uint8))
        Cr_binned = (np.floor(img_ycrcb[:,:,2]/divisor)).astype(np.uint8)
        
        # Find bin with most pixels
        most_pixels_bin = -1
        most_pixels = 0
        for bin_i in range(0, Cr_bins_n+1):
            n_pixels = np.count_nonzero(Cr_binned == bin_i)
            if n_pixels > most_pixels:
                most_pixels = n_pixels
                most_pixels_bin = bin_i
        
        # Find Stroma
        stroma_bin = most_pixels_bin
        stroma = Cr_binned == stroma_bin
        stroma = stroma + (Cr_binned == stroma_bin - 1)
        stroma = stroma + (Cr_binned == stroma_bin - 2)
        stroma = stroma * np.invert(background)
        stroma = morphology.dilation(stroma, morphology.square(3))
        stroma = morphology.remove_small_objects(stroma, 1000)
        
        # Find Epithelia
        epithelia_bin = stroma_bin + 2
        epithelia = Cr_binned == epithelia_bin
        epithelia = epithelia + (Cr_binned == epithelia_bin + 1)
        epithelia = epithelia + (Cr_binned == epithelia_bin + 2)
        epithelia = epithelia + (Cr_binned == epithelia_bin + 3)
        epithelia = epithelia + (Cr_binned == epithelia_bin + 4)
        epithelia = epithelia * np.invert(background)
        epithelia = epithelia * np.invert(stroma)
        epithelia = epithelia * np.invert(img_ycrcb[:,:,1] < 120)
        epithelia = morphology.dilation(epithelia, morphology.square(2))
        epithelia = morphology.remove_small_objects(epithelia, 500)
        epithelia = morphology.remove_small_holes(epithelia, 10000)
        
        # Adjust epithelia based on percentage
        n_epithelia_pixels = np.count_nonzero(epithelia)
        percent_epithelia_pixels = n_epithelia_pixels / (epithelia.shape[0] * epithelia.shape[1])
        small_obj_size_to_remove = 10000
        if percent_epithelia_pixels < 0.01:
            small_obj_size_to_remove = 1000
        elif percent_epithelia_pixels < 0.03:
            small_obj_size_to_remove = 2000
        
        epithelia = morphology.remove_small_objects(epithelia, small_obj_size_to_remove)
        epithelia = morphology.dilation(epithelia, morphology.square(10))
        epithelia = morphology.remove_small_holes(epithelia, 20000)
        
        # Apply masks
        epithelia_img = cv2.bitwise_and(img_rgb, img_rgb, mask=(epithelia.astype(np.uint8) * 255))
        
        # Expanded Stroma
        stroma = Cr_binned == stroma_bin + 1
        stroma = stroma + (Cr_binned == stroma_bin)
        stroma = stroma + (Cr_binned == stroma_bin - 1)
        stroma = stroma + (Cr_binned == stroma_bin - 2)
        stroma = stroma * np.invert(background)
        stroma = morphology.remove_small_holes(stroma, 10000)
        stroma = morphology.dilation(stroma, morphology.square(10))
        stroma = morphology.remove_small_holes(stroma, 20000)
        stroma = stroma * np.invert(background)
        stroma = stroma * np.invert(epithelia)
        
        stroma_img = cv2.bitwise_and(img_rgb, img_rgb, mask=(stroma.astype(np.uint8) * 255))
        
        return {
            'original': img_rgb,
            'stroma': stroma_img,
            'epithelia': epithelia_img
        }

    def display_segmentation_results(self, results, image_path):
        """Display segmentation results in the GUI"""
        # Create frame for this image's results
        image_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=f"Segmentation Results: {os.path.basename(image_path)}",
            padding=(10, 10)
        )
        image_frame.grid(row=self.current_row, column=0, pady=10, padx=10, sticky="ew")
        self.current_row += 1
        
        # Create horizontal frame for all images
        results_frame = ttk.Frame(image_frame)
        results_frame.pack(fill="x", expand=True)
        
        # Display each image with its label
        for i, (label, img) in enumerate([
            ("Original", results['original']),
            ("Stroma", results['stroma']),
            ("Epithelia", results['epithelia'])
        ]):
            # Create frame for this image
            img_frame = ttk.Frame(results_frame)
            img_frame.grid(row=0, column=i, padx=5, pady=5)
            
            # Add label
            ttk.Label(
                img_frame,
                text=label,
                font=('Arial', 12)
            ).pack(pady=(0, 5))
            
            # Resize and display image
            h, w = img.shape[:2]
            max_size = 300
            scale = max_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img_resized = cv2.resize(img, new_size)
            
            # Convert to PhotoImage
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Create and pack image label
            img_label = ttk.Label(img_frame, image=img_tk)
            img_label.image = img_tk
            img_label.pack()
            
            self.current_image_labels.append(img_label)
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # Update the select_files and select_folder methods to call process_segmentation:

    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[("TIF Files", "*.tif"), ("All Files", "*.*")]
        )
        if files:
            self.process_segmentation(files=files)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            self.process_segmentation(folder=folder)

    
    def select_he_folder(self):
        folder = filedialog.askdirectory(title="Select H&E Images Folder")
        if folder:
            self.he_folder = folder
            self.he_path_var.set(folder)
    
    def select_melan_folder(self):
        folder = filedialog.askdirectory(title="Select Melan Images Folder")
        if folder:
            self.melan_folder = folder
            self.melan_path_var.set(folder)
    
    def select_sox10_folder(self):
        folder = filedialog.askdirectory(title="Select Sox10 Images Folder")
        if folder:
            self.sox10_folder = folder
            self.sox10_path_var.set(folder)
    def clear_previous_results(self):
        # Clear all widgets in scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Reset current row
        self.current_row = 0
        
        # Clear image labels reference
        self.current_image_labels = []
        
        # Force garbage collection
        gc.collect()
    
    def process_matching(self):
        # Validate that at least two folders are selected
        selected_folders = [f for f in [self.he_folder, self.melan_folder, self.sox10_folder] if f]
        if len(selected_folders) < 2:
            messagebox.showerror("Error", "Please select at least two folders for matching")
            return
        
        # Clear previous results
        self.clear_previous_results()
        
        # Create a thread for processing
        thread = threading.Thread(target=self.run_matching_process)
        thread.start()
    
    def run_matching_process(self):
        try:
            self.progress_var.set("Processing: Loading images...")
            
            # Load images from folders
            patient_images = self.load_images_from_folders(
                self.he_folder,
                self.melan_folder,
                self.sox10_folder
            )
            
            # Process each patient
            for patient_id, raw_images in patient_images.items():
                self.progress_var.set(f"Processing patient: {patient_id}")
                
                # Extract tissue masks
                masked_images = self.extract_tissue_masks(raw_images)
                
                # Find matches
                similarity_scores = self.find_matches(raw_images, masked_images)
                
                # Display matches
                self.root.after(0, self.display_matches, similarity_scores, raw_images, patient_id)
            
            self.progress_var.set("Ready")
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Error in matching process: {str(e)}")
            self.progress_var.set("Ready")
    
    def load_images_from_folders(self, he_folder, melan_folder, sox10_folder):
        patient_images = {}
        
        if he_folder and os.path.exists(he_folder):
            for img_name in os.listdir(he_folder):
                if img_name.lower().endswith(('.tif', '.tiff')):
                    patient_id = img_name.split('_')[0]
                    if patient_id not in patient_images:
                        patient_images[patient_id] = {}
                    image_path = os.path.join(he_folder, img_name)
                    read_image = cv2.imread(image_path)
                    if read_image is not None:
                        patient_images[patient_id][f"h&e_{img_name}"] = read_image

        if melan_folder and os.path.exists(melan_folder):
            for img_name in os.listdir(melan_folder):
                if img_name.lower().endswith(('.tif', '.tiff')):
                    patient_id = img_name.split('_')[0]
                    if patient_id not in patient_images:
                        patient_images[patient_id] = {}
                    image_path = os.path.join(melan_folder, img_name)
                    read_image = cv2.imread(image_path)
                    if read_image is not None:
                        patient_images[patient_id][f"melan_{img_name}"] = read_image

        if sox10_folder and os.path.exists(sox10_folder):
            for img_name in os.listdir(sox10_folder):
                if img_name.lower().endswith(('.tif', '.tiff')):
                    patient_id = img_name.split('_')[0]
                    if patient_id not in patient_images:
                        patient_images[patient_id] = {}
                    image_path = os.path.join(sox10_folder, img_name)
                    read_image = cv2.imread(image_path)
                    if read_image is not None:
                        patient_images[patient_id][f"sox10_{img_name}"] = read_image
        
        return patient_images
    
    def extract_tissue_masks(self, raw_images):
        masked_images = {}

        for image_name, image in raw_images.items():
            # Converting to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            image_gray = cv2.equalizeHist(image_gray)
            
            # Blur to reduce noise
            blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
            
            # Adjust thresholding based on stain type
            if 'melan' in image_name:
                adaptive_thresholding = cv2.adaptiveThreshold(
                    blurred, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 1
                )
            elif 'sox10' in image_name:
                adaptive_thresholding = cv2.adaptiveThreshold(
                    blurred, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 13, 1
                )
            else:  # H&E
                adaptive_thresholding = cv2.adaptiveThreshold(
                    blurred, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 13, 1.8
                )
            
            # Create binary mask
            blurred_thresh = cv2.boxFilter(adaptive_thresholding, -1, (111, 111))
            mask_binary = blurred_thresh > 200
            mask_binary_image = (mask_binary.astype(np.uint8) * 255)
            
            masked_images[image_name] = mask_binary_image
        
        return masked_images
    
    def calculate_shape_similarity(self, image1, image2, image3=None):
        # Invert images
        image1 = cv2.bitwise_not(image1)
        image2 = cv2.bitwise_not(image2)
        if image3 is not None:
            image3 = cv2.bitwise_not(image3)
        
        # Find contours
        contours1, _ = cv2.findContours(image1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(image2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours3 = None
        if image3 is not None:
            contours3, _ = cv2.findContours(image3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours1 or not contours2:
            return None
        
        # Get largest contours
        sorted_contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)[:2]
        sorted_contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:2]
        sorted_contours3 = None
        if contours3:
            sorted_contours3 = sorted(contours3, key=cv2.contourArea, reverse=True)[:2]
        
        # Calculate similarity
        similarity = cv2.matchShapes(sorted_contours1[0], sorted_contours2[0], cv2.CONTOURS_MATCH_I1, 0.0)
        
        if sorted_contours3:
            similarity += cv2.matchShapes(sorted_contours1[0], sorted_contours3[0], cv2.CONTOURS_MATCH_I1, 0.0)
            similarity += cv2.matchShapes(sorted_contours2[0], sorted_contours3[0], cv2.CONTOURS_MATCH_I1, 0.0)
            similarity /= 3
        else:
            similarity /= 1
        
        return similarity

    def find_matches(self, raw_images, masked_images):
        similarity_scores = {}
        
        # Organize images by stain type
        he_images = {k: v for k, v in masked_images.items() if 'h&e' in k.lower()}
        melan_images = {k: v for k, v in masked_images.items() if 'melan' in k.lower()}
        sox10_images = {k: v for k, v in masked_images.items() if 'sox10' in k.lower()}
        
        # Check which stains are available
        available_stains = []
        if he_images:
            available_stains.append(('h&e', he_images))
        if melan_images:
            available_stains.append(('melan', melan_images))
        if sox10_images:
            available_stains.append(('sox10', sox10_images))
        
        # If we have all three stains
        if len(available_stains) == 3:
            for he_key, he_mask in he_images.items():
                for melan_key, melan_mask in melan_images.items():
                    for sox10_key, sox10_mask in sox10_images.items():
                        similarity = self.calculate_shape_similarity(he_mask, melan_mask, sox10_mask)
                        if similarity is not None and similarity < 0.6:
                            similarity_scores[(he_key, melan_key, sox10_key)] = similarity
        
        # If we have only two stains
        elif len(available_stains) == 2:
            stain1_name, stain1_images = available_stains[0]
            stain2_name, stain2_images = available_stains[1]
            
            for key1, mask1 in stain1_images.items():
                for key2, mask2 in stain2_images.items():
                    similarity = self.calculate_shape_similarity(mask1, mask2)
                    if similarity is not None and similarity < 0.6:
                        similarity_scores[(key1, key2)] = similarity
        
        return similarity_scores

    def setup_scrollable_frame(self):
        # Main container
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with both scrollbars
        self.canvas = tk.Canvas(self.canvas_frame)
        self.vsb = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.hsb = ttk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        
        # Create frame for content
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Configure canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(
            yscrollcommand=self.vsb.set,
            xscrollcommand=self.hsb.set
        )
        
        # Pack scrollbars and canvas
        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

    def display_matches(self, similarity_scores, raw_images, patient_id):
        if not similarity_scores:
            return

        # Create one row per patient
        patient_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=f"Matches for Patient {patient_id}",
            padding=(10, 10)
        )
        patient_frame.grid(row=self.current_row, column=0, pady=10, padx=10, sticky="ew")
        self.current_row += 1

        # Create horizontal frame for all match groups
        matches_frame = ttk.Frame(patient_frame)
        matches_frame.pack(fill="x", expand=True)

        # Display each match group horizontally
        for i, (group, score) in enumerate(sorted(similarity_scores.items(), key=lambda x: x[1])):
            # Create frame for this match group
            match_frame = ttk.LabelFrame(
                matches_frame,
                text=f"Match {i+1} (Score: {score:.4f})",
                padding=(5, 5)
            )
            match_frame.grid(row=0, column=i, padx=5, pady=5)

            # Create horizontal frame for images within this match
            images_frame = ttk.Frame(match_frame)
            images_frame.pack(fill="x", expand=True)

            # Display each image in the match group horizontally
            for j, name in enumerate(group):
                # Create frame for each image
                img_frame = ttk.Frame(images_frame)
                img_frame.grid(row=0, column=j, padx=2, pady=2)

                # Add image name
                ttk.Label(
                    img_frame,
                    text=Path(name).name,
                    font=('Arial', 10),
                    wraplength=150
                ).pack(pady=(0, 5))

                # Process and display image
                img = raw_images[name]
                h, w = img.shape[:2]
                max_size = 250  # Slightly smaller images for better horizontal fit
                scale = max_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                img_resized = cv2.resize(img, new_size)
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                
                # Create and pack image label
                img_label = ttk.Label(img_frame, image=img_tk)
                img_label.image = img_tk  # Keep a reference
                img_label.pack()

                self.current_image_labels.append(img_label)

        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
def main():
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()