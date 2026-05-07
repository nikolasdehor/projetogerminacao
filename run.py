"""Entry point — roda com: python run.py"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    print("\n🌱  Sistema de Monitoramento de Germinação")
    print("   Acesse: http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001, debug=True)
