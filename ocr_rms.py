import os
import glob
from PIL import Image
from pdf2image import convert_from_path
import pandas as pd
import pytesseract as pt
years = ['1955','1957','1958','1959','1960','1961','1962','1963','1964','1965','1966','1967','1968']

def to_csv(mayor_doc, tax_doc, counter):
    # Extract text from images using OCR
    mayor_text = ''
    tax_text = ''
    for img in mayor_doc:
        img = img.rotate(270, expand=True)
        mayor_text += pt.image_to_string(img)
    for img in tax_doc:
        img = img.rotate(270, expand=True)
        tax_text += pt.image_to_string(img)

    # Process mayor's document text
    mayor_lines = mayor_text.split('\n')
    mayor_data = [line.split() for line in mayor_lines if line.strip()]

    # Process tax document text
    tax_lines = tax_text.split('\n')
    tax_data = [line.split() for line in tax_lines if line.strip()]

    # Convert to DataFrames
    mayor_df = pd.DataFrame(mayor_data[1:])
    tax_df = pd.DataFrame(tax_data[1:])

    # Save to CSV files
    mayor_df.to_csv('mayor_data' + years[counter] + '.csv', index=False)
    tax_df.to_csv('tax_data' + years[counter] + '.csv', index=False)



current_dir = os.path.dirname(os.path.abspath(__file__))
mayors = os.path.join(current_dir, 'mayors')
taxes = os.path.join(current_dir, 'taxes')

for i in range(0, min(len(os.listdir(mayors)), len(os.listdir(taxes)))):
    mayor_pdfs = sorted(glob.glob(os.path.join(mayors, '*.pdf')))
    tax_pdfs = sorted(glob.glob(os.path.join(taxes, '*.pdf')))
    to_csv(convert_from_path(mayor_pdfs[i], dpi=300, poppler_path=r"C:\Poppler\poppler-25.07.0\Library\bin"), convert_from_path(tax_pdfs[i], dpi=300, poppler_path=r"C:\Poppler\poppler-25.07.0\Library\bin"), i)