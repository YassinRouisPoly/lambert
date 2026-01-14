echo "Préparation des dossiers"
mkdir -p datasets/original models/ cache/
cd datasets/original
echo "Téléchargement du dataset cold-french-law depuis HuggingFace ..."
curl -OL https://huggingface.co/datasets/harvard-lil/cold-french-law/resolve/main/cold-french-law.csv?download=true
echo "Terminé"