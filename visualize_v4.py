def show_batch(self, event=None):
    print("\n" + "="*50)
    print("LOADING NEW BATCH")
    print("="*50)

    # Clear axes first
    for ax in self.axs:
        ax.clear()
        ax.axis('off')

    # Track if any annotation errors occur
    annotation_error = None

    # Get batch of images
    batch = self.get_random_batch()
    if not batch:
        self.info_text.set_text("No items/images to display!")
        self.fig.canvas.draw_idle()
        return

    for i, (source, ax) in enumerate(zip(batch, self.axs)):
        img = self.load_image(source)
        if img is not None:
            ax.imshow(img)
            # Draw annotations only if not in directory mode
            if self.dataset_format != "directory":
                try:
                    self.draw_annotations(ax, source)
                except Exception as e:
                    annotation_error = e  # save error to show globally
            # Set title
            title = f"Image {i+1}" if self.dataset_format != "directory" else os.path.basename(source)
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax.text(0.5,0.5,"Failed to load image",ha='center',va='center',color='red',fontsize=14)
        ax.axis('off')

    # Hide extra axes
    for i in range(len(batch), len(self.axs)):
        self.axs[i].set_visible(False)

    # Info text
    info = f"Showing {len(batch)} images | Total: {len(self.items) or len(self.image_files)}"
    if self.dataset_format != "directory":
        info += f" | Classes: {len(self.class_names)}"

    # --- SHOW GLOBAL WARNING IF ANNOTATIONS FAILED ---
    if annotation_error:
        warning_msg = f"⚠ Warning: annotations failed to load: {annotation_error}"
        # Show warning at top-left of figure (not tied to any subplot)
        self.fig.text(
            0.02, 0.95, warning_msg,
            fontsize=10, color='orange',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange', alpha=0.8)
        )

    self.info_text.set_text(info)
    self.fig.canvas.draw_idle()
    print(f"✓ Displayed batch of {len(batch)} images")














def show_batch(self, event=None):
    print("\n" + "="*50)
    print("LOADING NEW BATCH")
    print("="*50)

    # Clear axes first
    for ax in self.axs:
        ax.clear()
        ax.axis('off')

    # Get batch of images
    batch = self.get_random_batch()
    if not batch:
        self.info_text.set_text("No items/images to display!")
        self.fig.canvas.draw_idle()
        return

    for i, (source, ax) in enumerate(zip(batch, self.axs)):
        img = self.load_image(source)
        if img is not None:
            ax.imshow(img)
            # Draw annotations only if we are not in directory mode
            if self.dataset_format != "directory":
                try:
                    self.draw_annotations(ax, source)
                except Exception as e:
                    # Capture annotation drawing errors
                    warn_msg = f"⚠ Annotation load failed: {e}"
                    ax.text(0.5, 0.95, warn_msg, ha='center', va='top', fontsize=9,
                            color='orange', transform=ax.transAxes,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='orange', alpha=0.8))
            # Set title
            title = f"Image {i+1}" if self.dataset_format != "directory" else os.path.basename(source)
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax.text(0.5,0.5,"Failed to load image",ha='center',va='center',color='red',fontsize=14)
        ax.axis('off')

    # Hide extra axes
    for i in range(len(batch), len(self.axs)):
        self.axs[i].set_visible(False)

    # Info text
    info = f"Showing {len(batch)} images | Total: {len(self.items) or len(self.image_files)}"
    if self.dataset_format != "directory":
        info += f" | Classes: {len(self.class_names)}"
    self.info_text.set_text(info)

    self.fig.canvas.draw_idle()
    print(f"Displayed batch of {len(batch)} images")






















def _load_class_names(self):
    try:
        categories = self.dataset.categories()
        if 'label' in categories:
            label_cat = categories['label']
            return [label_cat.items[i].name for i in range(len(label_cat.items))]
    except:
        pass

    # Fallback: create as many classes as needed dynamically
    max_label = 0
    for item in self.items:
        for anno in getattr(item, 'annotations', []):
            if hasattr(anno, 'label'):
                max_label = max(max_label, anno.label)

    return [f"Class_{i}" for i in range(max_label + 1)]
