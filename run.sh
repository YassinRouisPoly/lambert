docker compose up --build &
cd server
uvicorn main:app &
cd ../client
python3 -m http.server 80