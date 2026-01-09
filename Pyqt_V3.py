# In RenderThread.run() - simpler alternative
def run(self):
    images = []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    for item in self.items:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Try to get the image data
            if hasattr(item, 'image'):
                img_data = None
                if hasattr(item.image, 'data'):
                    img_data = item.image.data
                elif hasattr(item.image, 'path'):
                    import cv2
                    img = cv2.imread(item.image.path)
                    if img is not None:
                        img_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if img_data is not None:
                    ax.imshow(img_data)
            
            # Try to draw annotations if available
            if hasattr(item, 'annotations'):
                # Simple annotation drawing
                for ann in item.annotations:
                    if hasattr(ann, 'points'):
                        # Draw bounding box
                        points = ann.points
                        if len(points) == 4:  # xmin, ymin, xmax, ymax
                            import matplotlib.patches as patches
                            rect = patches.Rectangle(
                                (points[0], points[1]), 
                                points[2] - points[0], 
                                points[3] - points[1],
                                linewidth=2, 
                                edgecolor='r', 
                                facecolor='none'
                            )
                            ax.add_patch(rect)
            
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            # Convert to PIL
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(Image.fromarray(img_array))
            
            plt.close(fig)
            
        except Exception as e:
            # Create a blank image as fallback
            blank_img = Image.new('RGB', (400, 300), color='gray')
            images.append(blank_img)
            plt.close('all')
    
    self.finished.emit(images)
