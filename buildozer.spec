[app]
title = Smart Document Scanner
package.name = smartdocscanner
package.domain = org.smartscanner

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,ttf,atlas
source.include_patterns = assets/fonts/*.ttf,scanner_core/*,scanner_core/**/*

version = 1.0.0

# Python + native libraries needed at runtime.
# opencv-python (the "opencv" p4a recipe) already bundles numpy.
requirements = python3,kivy==2.3.1,opencv,numpy,pillow,img2pdf,plyer,android

# Portrait phone app.
orientation = portrait
fullscreen = 0

icon.filename = %(source.dir)s/assets/icon.png

# ---------------------------------------------------------------------
# Android specific
# ---------------------------------------------------------------------
android.permissions = CAMERA,READ_MEDIA_IMAGES,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.api = 34
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a,armeabi-v7a
android.allow_backup = True

# Needed because the app reads/writes files (loaded photos, saved PDFs).
android.add_src =

[buildozer]
log_level = 2
warn_on_root = 1
