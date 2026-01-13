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
