import zipfile
import xml.etree.ElementTree as ET

def extract_code_from_mlx(mlx_path, output_m_path=None):
    # 1️⃣ Ouvre le .mlx comme archive
    with zipfile.ZipFile(mlx_path, 'r') as zip_ref:
        # 2️⃣ Cherche le fichier document.xml
        xml_name = None
        for name in zip_ref.namelist():
            if name.endswith('document.xml'):
                xml_name = name
                break
        if xml_name is None:
            raise FileNotFoundError("Impossible de trouver document.xml dans l'archive .mlx")
        
        with zip_ref.open(xml_name) as f:
            xml_content = f.read()

    # 3️⃣ Parse le XML
    root = ET.fromstring(xml_content)

    # 4️⃣ Extraire toutes les lignes de code MATLAB
    code_lines = []
    for elem in root.iter():
        tag = elem.tag.lower()
        # On cible les lignes de code MATLAB
        if tag.endswith('line') or tag.endswith('code'):
            text = (elem.text or '').strip()
            if text:
                code_lines.append(text)

    full_code = "\n".join(code_lines)

    # 5️⃣ Sauvegarde ou affiche
    if output_m_path:
        with open(output_m_path, 'w', encoding='utf-8') as f:
            f.write(full_code)
        print(f"✅ Code extrait et sauvegardé dans {output_m_path}")
    else:
        print(full_code)

    # 6️⃣ Feedback
    if not full_code.strip():
        print("⚠️ Aucune ligne de code détectée. Le format XML est peut-être plus complexe.")
    else:
        print(f"📄 {len(code_lines)} lignes de code extraites.")

    return full_code


# Exemple d'utilisation :
#extract_code_from_mlx(
#    "cellsize_sphere_PGSE_with_dictionary_creation_V2.mlx",
#    "cellsize_sphere_PGSE_with_dictionary_creation_V2.m"
#)

import zipfile

with zipfile.ZipFile("cellsize_sphere_PGSE_with_dictionary_creation_V2.mlx", 'r') as z:
    with z.open("matlab/document.xml") as f:
        for i, line in enumerate(f):
            if i > 20:  # affiche seulement les 20 premières lignes
                break
            print(line.decode('utf-8').strip())



#import zipfile
#z = zipfile.ZipFile("cellsize_sphere_PGSE_with_dictionary_creation_V2.mlx")
#print(z.namelist())
