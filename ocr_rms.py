import os
import glob
import cv2
import pikepdf as pk
import pytesseract as pt
import numpy as np
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path
from PyPDF2 import PdfMerger
import pandas as pd
import pytesseract as pt
years = ['1955','1957','1958','1959','1960','1961','1962','1963','1964','1965','1966','1967','1968']

def enhance_image(img) -> Image:
    img = img.convert('RGB') #ensure image is in RGB mode
    img = np.array(img) #convert PIL image to numpy array
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC) #resize image to improve ocr accuracy
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #convert to grayscale
    img = cv2.GaussianBlur(img, (5, 5), 0) #apply gaussian blur to reduce noise
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #apply adaptive thresholding to enhance text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    img = cv2.morphologyEx(img[1], cv2.MORPH_OPEN, kernel) #apply morphological operations to enhance text
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #apply morphological operations to enhance text
    img = Image.fromarray(img.astype('uint8')) #convert back to PIL image
    return img

def to_csv(mayor_doc, tax_doc, counter):
    merger = PdfMerger()
    for img in mayor_doc:
        img = img.rotate(270, expand=True)
        img = enhance_image(img) #improve image quality for ocr
        searchable_mayors = pt.image_to_pdf_or_hocr(img, extension='pdf')
        buf = BytesIO(searchable_mayors)
        buf.seek(0)
        merger.append(buf)
    for img in tax_doc:
        if counter < 12: #for years before 1967, rotate tax images to be upright
            img = img.rotate(270, expand=True)
        img = enhance_image(img) #improve image quality for ocr
        searchable_taxes = pt.image_to_pdf_or_hocr(img, extension='pdf')
        buf = BytesIO(searchable_taxes)
        buf.seek(0)
        merger.append(buf)

    merger.write(f'ocr_rms_{years[counter]}.pdf')
    merger.close()
    with pk.open('ocr_rms_{years[counter]}.pdf') as pdf:
        pdf.save('ocr_rms_{years[counter]}.pdf', optimize_image=True)


current_dir = os.path.dirname(os.path.abspath(__file__))
mayors = os.path.join(current_dir, 'mayors')
taxes = os.path.join(current_dir, 'taxes')

for i in range(0, min(len(os.listdir(mayors)), len(os.listdir(taxes)))):
    mayor_pdfs = sorted(glob.glob(os.path.join(mayors, '*.pdf')))
    tax_pdfs = sorted(glob.glob(os.path.join(taxes, '*.pdf')))
    to_csv(convert_from_path(mayor_pdfs[i], dpi=300, poppler_path=r"C:\Poppler\poppler-25.07.0\Library\bin"), convert_from_path(tax_pdfs[i], dpi=300, poppler_path=r"C:\Poppler\poppler-25.07.0\Library\bin"), i)