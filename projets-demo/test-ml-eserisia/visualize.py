import matplotlib.pyplot as plt
import pandas as pd
import torch
from main import EserisiaModel

def visualize_results():
    """Visualise les résultats du modèle"""
    print("📊 Visualisation des résultats ESERISIA AI")
    
    # Charger les données
    try:
        data = pd.read_csv('data.csv')
        
        # Graphique des données
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(data['target'], bins=30, alpha=0.7, color='blue')
        plt.title('Distribution de la variable cible')
        plt.xlabel('Valeur')
        plt.ylabel('Fréquence')
        
        plt.subplot(1, 2, 2)
        correlation_matrix = data.corr()
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title('Matrice de corrélation')
        plt.xticks(range(len(data.columns)), data.columns, rotation=45)
        plt.yticks(range(len(data.columns)), data.columns)
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Graphiques sauvegardés: data_analysis.png")
        
    except Exception as e:
        print(f"❌ Erreur lors de la visualisation: {e}")

if __name__ == "__main__":
    visualize_results()
