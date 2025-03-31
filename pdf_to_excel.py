import pandas as pd
import tabula
import os

def convert_pdf_to_excel(pdf_path, excel_path):
    """
    Convert PDF table to Excel format
    """
    try:
        # Read all tables from the PDF
        print("Reading PDF file...")
        tables = tabula.read_pdf(pdf_path, pages='all')
        
        if not tables:
            print("No tables found in the PDF")
            return False
            
        # Combine all tables if there are multiple
        df = pd.concat(tables, ignore_index=True)
        
        # Save to Excel
        print("Saving to Excel...")
        df.to_excel(excel_path, index=False)
        print(f"Successfully saved to {excel_path}")
        return True
        
    except Exception as e:
        print(f"Error converting PDF to Excel: {e}")
        return False

if __name__ == "__main__":
    pdf_path = "soildataset.xlsx - Google Sheets.pdf"
    excel_path = "soildataset.xlsx"
    
    if convert_pdf_to_excel(pdf_path, excel_path):
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!") 