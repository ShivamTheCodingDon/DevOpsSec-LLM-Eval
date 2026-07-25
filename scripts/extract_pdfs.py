import fitz
import os

papers_dir = r"D:\000Coding\ACM"
output_dir = r"D:\000Coding\ACM\paper_texts"
os.makedirs(output_dir, exist_ok=True)

pdf_files = [f for f in os.listdir(papers_dir) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(papers_dir, pdf_file)
    txt_name = pdf_file.replace('.pdf', '.txt')
    txt_path = os.path.join(output_dir, txt_name)
    
    print(f"\nProcessing: {pdf_file}")
    
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        full_text = ""
        for i in range(num_pages):
            page = doc[i]
            page_text = page.get_text()
            full_text += f"\n--- PAGE {i+1} ---\n{page_text}"
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        doc.close()
        print(f"  Pages: {num_pages}, Chars: {len(full_text)}, Saved: {txt_path}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

print("\nDone!")
