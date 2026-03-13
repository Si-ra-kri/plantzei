"""
training/train_disease.py

Trains a MobileNetV2 transfer-learning CNN for plant disease classification.

PREREQUISITES:
  1. Install TensorFlow:  pip install tensorflow
  2. Download PlantVillage dataset from Kaggle:
     https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
  3. Set DATASET_DIR below to the extracted path, which should contain
     subdirectories named by disease class (e.g., Tomato___healthy, etc.)
  4. Run: python training/train_disease.py

The script remaps PlantVillage subdirectory names to our 6 classes:
  Healthy, Fungi/Blight, Bacterial Spot, Insect Damage, Rust, Mosaic Virus

Saves model to: models/disease_model.h5
"""

import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR = os.environ.get("PLANT_VILLAGE_DIR", "./data/plantvillage")  # override via env
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "..", "models", "disease_model.h5")
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

# ── Class remapping (PlantVillage folder → our 6 classes) ─────────────────────
CLASS_REMAP = {
    "healthy":             "Healthy",
    "early_blight":        "Fungi/Blight",
    "late_blight":         "Fungi/Blight",
    "leaf_blight":         "Fungi/Blight",
    "gray_leaf_spot":      "Fungi/Blight",
    "bacterial_spot":      "Bacterial Spot",
    "spider_mites":        "Insect Damage",
    "leaf_miner":          "Insect Damage",
    "rust":                "Rust",
    "leaf_rust":           "Rust",
    "mosaic_virus":        "Mosaic Virus",
    "yellow_leaf_curl_virus": "Mosaic Virus",
}
TARGET_CLASSES = ["Healthy", "Fungi/Blight", "Bacterial Spot", "Insect Damage", "Rust", "Mosaic Virus"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def remap_directory(src_dir: str, dst_dir: str):
    """Copy/symlink PlantVillage images into remapped class subdirectories."""
    import shutil
    os.makedirs(dst_dir, exist_ok=True)
    for cls in TARGET_CLASSES:
        os.makedirs(os.path.join(dst_dir, cls), exist_ok=True)

    for folder in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        # Find matching remap key
        folder_lower = folder.lower()
        target = "Healthy"  # default
        for key, mapped in CLASS_REMAP.items():
            if key in folder_lower:
                target = mapped
                break
        dst_folder = os.path.join(dst_dir, target)
        for img_file in os.listdir(folder_path):
            src = os.path.join(folder_path, img_file)
            dst = os.path.join(dst_folder, f"{folder}_{img_file}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    print(f"✅ Dataset remapped to {dst_dir}")


def train():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        print("❌ TensorFlow not installed. Run: pip install tensorflow")
        sys.exit(1)

    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset not found at: {DATASET_DIR}")
        print("   Download from: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("   Then set PLANT_VILLAGE_DIR env variable or edit DATASET_DIR in this script.")
        sys.exit(1)

    # Remap to canonical class dirs
    remapped_dir = "./data/plantvillage_remapped"
    print("📁 Remapping dataset classes …")
    remap_directory(DATASET_DIR, remapped_dir)

    # Data generators with augmentation
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
    )

    train_gen = datagen.flow_from_directory(
        remapped_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training",
    )
    val_gen = datagen.flow_from_directory(
        remapped_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation",
    )

    n_classes = len(train_gen.class_indices)
    print(f"📊 Classes: {list(train_gen.class_indices.keys())}")

    # ── Build MobileNetV2 transfer-learning model ──────────────────────────────
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False  # freeze base

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)

    print("🔧 Phase 1 — Training classification head (backbone frozen) …")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=5)

    # ── Fine-tune last 20 layers ───────────────────────────────────────────────
    print("🔧 Phase 2 — Fine-tuning last 20 layers …")
    for layer in base.layers[-20:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS - 5)

    model.save(MODEL_OUT)
    print(f"💾 Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    train()
