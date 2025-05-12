import tensorflow_datasets as tfds

# تحميل معلومات الداتا فقط بدون تحميل الصور
ds_info = tfds.builder("oxford_iiit_pet").info

# الحصول على أسماء الفصائل (labels)
label_names = ds_info.features["label"].names

# طباعة الفصائل
print("📌 All Breeds in Oxford-IIIT Pet Dataset:\n")
for i, name in enumerate(label_names):
    print(f"{i:2d} → {name}")
