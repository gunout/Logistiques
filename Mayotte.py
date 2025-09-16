import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortMayotteTradeAnalyzer:
    def __init__(self):
        self.port_name = "Port de Longoni - Mayotte"
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', 
                      '#AB83A1', '#5CAB7D', '#2A9D8F', '#E76F51', '#264653']
        
        self.start_year = 2010
        self.end_year = 2024
        
    def generate_trade_data(self):
        """Génère des données d'import-export pour le Port de Longoni"""
        print("📊 Génération des données de commerce pour le Port de Longoni...")
        
        # Créer une base de données mensuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='M')
        
        data = {'Date': dates}
        
        # Données d'importation (en tonnes)
        data['Import_Tonnes'] = self._simulate_imports(dates)
        data['Import_Valeur_EUR'] = data['Import_Tonnes'] * np.random.uniform(800, 1200, len(dates))
        
        # Données d'exportation (en tonnes)
        data['Export_Tonnes'] = self._simulate_exports(dates)
        data['Export_Valeur_EUR'] = data['Export_Tonnes'] * np.random.uniform(500, 800, len(dates))
        
        # Balance commerciale
        data['Balance_Commerciale'] = data['Export_Valeur_EUR'] - data['Import_Valeur_EUR']
        
        # Principales catégories de marchandises (spécifiques à Mayotte)
        data['Produits_Alimentaires_Tonnes'] = self._simulate_food_imports(dates)
        data['Materiaux_Construction_Tonnes'] = self._simulate_construction_materials(dates)
        data['Vehicules_Tonnes'] = self._simulate_vehicles_imports(dates)
        data['Produits_Petroliers_Tonnes'] = self._simulate_petroleum_imports(dates)
        data['Ylang_Ylang_Tonnes'] = self._simulate_ylang_ylang_exports(dates)
        data['Vanille_Tonnes'] = self._simulate_vanilla_exports(dates)
        
        # Principaux partenaires commerciaux
        data['France_Import_Tonnes'] = self._simulate_france_trade(dates)
        data['Madagascar_Import_Tonnes'] = self._simulate_madagascar_trade(dates)
        data['Comores_Import_Tonnes'] = self._simulate_comoros_trade(dates)
        data['Afrique_Import_Tonnes'] = self._simulate_africa_trade(dates)
        
        # Activité portuaire
        data['Navires_Entrees'] = self._simulate_ship_arrivals(dates)
        data['Conteneurs_Traites'] = self._simulate_containers(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances saisonnières et économiques
        self._add_economic_trends(df)
        
        return df
    
    def _simulate_imports(self, dates):
        """Simule les importations du Port de Longoni"""
        base_imports = 40000  # tonnes/mois de base (dépendance forte aux importations)
        
        imports = []
        for i, date in enumerate(dates):
            # Croissance annuelle de 4-6% (forte croissance démographique)
            growth = 1 + (0.04 + 0.02 * (i / len(dates)))
            
            # Saisonnalité (pic avant Ramadan et fêtes)
            month = date.month
            if month in [4, 5]:  # Pic avant Ramadan
                seasonal = 1.3
            elif month in [11, 12]:  # Période de fêtes
                seasonal = 1.2
            else:
                seasonal = 1.0
            
            # Bruit aléatoire
            noise = np.random.normal(1, 0.12)
            
            imports.append(base_imports * growth * seasonal * noise)
        
        return imports
    
    def _simulate_exports(self, dates):
        """Simule les exportations du Port de Longoni"""
        base_exports = 5000  # tonnes/mois de base (faible volume d'exportations)
        
        exports = []
        for i, date in enumerate(dates):
            # Croissance modeste
            growth = 1 + (0.01 + 0.01 * (i / len(dates)))
            
            # Saisonnalité des produits agricoles
            month = date.month
            if month in [6, 7, 8]:  # Pic production ylang-ylang
                seasonal = 1.5
            elif month in [1, 2]:  # Saison des pluies
                seasonal = 0.7
            else:
                seasonal = 1.0
            
            noise = np.random.normal(1, 0.2)  # Plus volatile
            
            exports.append(base_exports * growth * seasonal * noise)
        
        return exports
    
    def _simulate_food_imports(self, dates):
        """Simule les importations de produits alimentaires"""
        base_volume = 15000
        
        volumes = []
        for date in dates:
            # Pic avant Ramadan et fêtes
            month = date.month
            if month in [4, 5, 11, 12]:
                seasonal = 1.4
            else:
                seasonal = 1.0
            
            # Forte croissance liée à la démographie
            growth = 1 + 0.05 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.1)
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_construction_materials(self, dates):
        """Simule les matériaux de construction"""
        base_volume = 10000
        
        volumes = []
        for date in dates:
            # Croissance forte due au développement infrastructurel
            growth = 1 + 0.06 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_vehicles_imports(self, dates):
        """Simule l'importation de véhicules"""
        base_volume = 2000
        
        volumes = []
        for date in dates:
            # Croissance modérée
            growth = 1 + 0.03 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.2)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_petroleum_imports(self, dates):
        """Simule les importations de produits pétroliers"""
        base_volume = 8000
        
        volumes = []
        for date in dates:
            # Croissance régulière
            growth = 1 + 0.02 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.1)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_ylang_ylang_exports(self, dates):
        """Simule les exportations d'ylang-ylang (produit phare de Mayotte)"""
        base_volume = 500
        
        volumes = []
        for date in dates:
            # Saisonnalité forte (floraison)
            month = date.month
            if 6 <= month <= 9:  # Période de récolte principale
                seasonal = 1.8
            else:
                seasonal = 0.5
            
            # Croissance modeste
            growth = 1 + 0.02 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.25)
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_vanilla_exports(self, dates):
        """Simule les exportations de vanille"""
        base_volume = 200
        
        volumes = []
        for date in dates:
            # Saisonnalité
            month = date.month
            if 7 <= month <= 10:  # Période de récolte
                seasonal = 1.6
            else:
                seasonal = 0.6
            
            # Croissance modérée
            growth = 1 + 0.03 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.3)
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_france_trade(self, dates):
        """Simule le commerce avec la France métropolitaine"""
        base_volume = 25000
        
        volumes = []
        for date in dates:
            # Relation stable avec la métropole
            growth = 1 + 0.03 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.08)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_madagascar_trade(self, dates):
        """Simule le commerce avec Madagascar"""
        base_volume = 5000
        
        volumes = []
        for date in dates:
            # Croissance modérée
            growth = 1 + 0.04 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.2)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_comoros_trade(self, dates):
        """Simule le commerce avec les Comores"""
        base_volume = 3000
        
        volumes = []
        for date in dates:
            # Commerce volatile
            growth = 1 + 0.02 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.25)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_africa_trade(self, dates):
        """Simule le commerce avec l'Afrique continentale"""
        base_volume = 2000
        
        volumes = []
        for date in dates:
            # Croissance modérée
            growth = 1 + 0.03 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_ship_arrivals(self, dates):
        """Simule le nombre de navires entrants"""
        base_arrivals = 80
        
        arrivals = []
        for date in dates:
            # Saisonnalité maritime
            month = date.month
            if month in [5, 6, 9, 10]:  # Meilleures conditions météo
                seasonal = 1.1
            elif month in [1, 2]:  # Saison cyclonique
                seasonal = 0.8
            else:
                seasonal = 1.0
            
            growth = 1 + 0.02 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.1)
            arrivals.append(base_arrivals * growth * seasonal * noise)
        
        return arrivals
    
    def _simulate_containers(self, dates):
        """Simule le nombre de conteneurs traités"""
        base_containers = 15000
        
        containers = []
        for date in dates:
            growth = 1 + 0.04 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.12)
            containers.append(base_containers * growth * noise)
        
        return containers
    
    def _add_economic_trends(self, df):
        """Ajoute des tendances économiques réalistes"""
        for i, row in df.iterrows():
            date = row['Date']
            year = date.year
            
            # Impact COVID-19 (2020-2021)
            if 2020 <= year <= 2021:
                if year == 2020 and date.month in [3, 4, 5]:
                    df.loc[i, 'Import_Tonnes'] *= 0.7
                    df.loc[i, 'Export_Tonnes'] *= 0.6
                    df.loc[i, 'Navires_Entrees'] *= 0.8
            
            # Croissance économique post-COVID
            elif year >= 2022:
                recovery = 1 + 0.03 * (year - 2022)
                df.loc[i, 'Import_Tonnes'] *= recovery
                df.loc[i, 'Export_Tonnes'] *= recovery
            
            # Développement des infrastructures (2014-2018)
            elif 2014 <= year <= 2018:
                df.loc[i, 'Materiaux_Construction_Tonnes'] *= 1.15
            
            # Augmentation de la population (forte croissance démographique)
            if year >= 2011:
                pop_growth = 1 + 0.02 * (year - 2011)
                df.loc[i, 'Produits_Alimentaires_Tonnes'] *= pop_growth
    
    def create_trade_analysis(self, df):
        """Crée une analyse complète du commerce portuaire"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Importations vs Exportations
        ax1 = plt.subplot(3, 2, 1)
        self._plot_trade_balance(df, ax1)
        
        # 2. Types de marchandises
        ax2 = plt.subplot(3, 2, 2)
        self._plot_commodity_types(df, ax2)
        
        # 3. Partenaires commerciaux
        ax3 = plt.subplot(3, 2, 3)
        self._plot_trade_partners(df, ax3)
        
        # 4. Activité portuaire
        ax4 = plt.subplot(3, 2, 4)
        self._plot_port_activity(df, ax4)
        
        # 5. Évolution annuelle
        ax5 = plt.subplot(3, 2, 5)
        self._plot_yearly_evolution(df, ax5)
        
        # 6. Analyse saisonnière
        ax6 = plt.subplot(3, 2, 6)
        self._plot_seasonal_analysis(df, ax6)
        
        plt.suptitle(f'Analyse Import-Export - {self.port_name} (2010-2024)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('port_mayotte_trade_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Générer les insights
        self._generate_trade_insights(df)
    
    def _plot_trade_balance(self, df, ax):
        """Plot de la balance commerciale"""
        ax.plot(df['Date'], df['Import_Tonnes']/1000, label='Importations', 
               linewidth=2, color='red', alpha=0.8)
        ax.plot(df['Date'], df['Export_Tonnes']/1000, label='Exportations', 
               linewidth=2, color='green', alpha=0.8)
        
        ax.set_title('Importations vs Exportations (milliers de tonnes)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Tonnes (x1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ajouter la balance commerciale en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Date'], df['Balance_Commerciale']/1e6, 
                label='Balance Commerciale', linewidth=1, color='blue', linestyle='--')
        ax2.set_ylabel('Balance (M€)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
    
    def _plot_commodity_types(self, df, ax):
        """Plot des types de marchandises"""
        commodities = ['Produits_Alimentaires_Tonnes', 'Materiaux_Construction_Tonnes', 
                      'Vehicules_Tonnes', 'Produits_Petroliers_Tonnes',
                      'Ylang_Ylang_Tonnes', 'Vanille_Tonnes']
        labels = ['Produits Alimentaires', 'Matériaux Construction', 'Véhicules', 
                 'Produits Pétroliers', 'Ylang-Ylang', 'Vanille']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', '#AB83A1']
        
        for i, col in enumerate(commodities):
            ax.plot(df['Date'], df[col]/1000, label=labels[i], 
                   linewidth=2, color=colors[i], alpha=0.8)
        
        ax.set_title('Types de Marchandises (milliers de tonnes)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Tonnes (x1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_trade_partners(self, df, ax):
        """Plot des partenaires commerciaux"""
        partners = ['France_Import_Tonnes', 'Madagascar_Import_Tonnes', 
                   'Comores_Import_Tonnes', 'Afrique_Import_Tonnes']
        labels = ['France', 'Madagascar', 'Comores', 'Afrique']
        colors = ['#264653', '#2A9D8F', '#E76F51', '#F9A602']
        
        for i, col in enumerate(partners):
            ax.plot(df['Date'], df[col]/1000, label=labels[i], 
                   linewidth=2, color=colors[i], alpha=0.8)
        
        ax.set_title('Importations par Partenaire Commercial (milliers de tonnes)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Tonnes (x1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_port_activity(self, df, ax):
        """Plot de l'activité portuaire"""
        ax.plot(df['Date'], df['Navires_Entrees'], label='Navires Entrants', 
               linewidth=2, color='purple', alpha=0.8)
        
        ax2 = ax.twinx()
        ax2.plot(df['Date'], df['Conteneurs_Traites']/1000, 
                label='Conteneurs Traités (x1000)', linewidth=2, color='orange', alpha=0.8)
        
        ax.set_title('Activité Portuaire', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nombre de Navires', color='purple')
        ax2.set_ylabel('Conteneurs (x1000)', color='orange')
        ax.tick_params(axis='y', labelcolor='purple')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.grid(True, alpha=0.3)
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_yearly_evolution(self, df, ax):
        """Plot de l'évolution annuelle"""
        df_yearly = df.copy()
        df_yearly['Year'] = df_yearly['Date'].dt.year
        
        yearly_data = df_yearly.groupby('Year').agg({
            'Import_Tonnes': 'mean',
            'Export_Tonnes': 'mean',
            'Balance_Commerciale': 'mean'
        })
        
        x = yearly_data.index
        width = 0.35
        
        ax.bar(x - width/2, yearly_data['Import_Tonnes']/1000, width, 
               label='Importations', color='red', alpha=0.7)
        ax.bar(x + width/2, yearly_data['Export_Tonnes']/1000, width, 
               label='Exportations', color='green', alpha=0.7)
        
        ax.set_title('Évolution Annuelle Moyenne (milliers de tonnes)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Tonnes (x1000)')
        ax.set_xlabel('Année')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_seasonal_analysis(self, df, ax):
        """Analyse saisonnière"""
        df_seasonal = df.copy()
        df_seasonal['Month'] = df_seasonal['Date'].dt.month
        
        seasonal_data = df_seasonal.groupby('Month').agg({
            'Import_Tonnes': 'mean',
            'Export_Tonnes': 'mean'
        })
        
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        
        ax.plot(months, seasonal_data['Import_Tonnes']/1000, 
               label='Importations', marker='o', color='red')
        ax.plot(months, seasonal_data['Export_Tonnes']/1000, 
               label='Exportations', marker='o', color='green')
        
        ax.set_title('Analyse Saisonnière (moyenne mensuelle)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Tonnes (x1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    def _generate_trade_insights(self, df):
        """Génère des insights analytiques"""
        print(f"📊 INSIGHTS ANALYTIQUES - {self.port_name}")
        print("=" * 60)
        
        # 1. Statistiques de base
        print("\n1. 📈 STATISTIQUES GÉNÉRALES (2010-2024):")
        total_imports = df['Import_Tonnes'].sum() / 1e6
        total_exports = df['Export_Tonnes'].sum() / 1e6
        trade_balance = df['Balance_Commerciale'].mean() / 1e6
        
        print(f"Importations totales: {total_imports:.1f} millions de tonnes")
        print(f"Exportations totales: {total_exports:.1f} millions de tonnes")
        print(f"Balance commerciale moyenne: {trade_balance:.1f} M€/mois")
        
        # 2. Croissance
        print("\n2. 📊 TAUX DE CROISSANCE:")
        growth_imports = ((df['Import_Tonnes'].iloc[-12:].mean() / 
                          df['Import_Tonnes'].iloc[:12].mean()) - 1) * 100
        growth_exports = ((df['Export_Tonnes'].iloc[-12:].mean() / 
                          df['Export_Tonnes'].iloc[:12].mean()) - 1) * 100
        
        print(f"Croissance des importations: {growth_imports:.1f}%")
        print(f"Croissance des exportations: {growth_exports:.1f}%")
        
        # 3. Principales marchandises
        print("\n3. 🚢 RÉPARTITION DES MARCHANDISES:")
        commodities = {
            'Produits Alimentaires': df['Produits_Alimentaires_Tonnes'].mean(),
            'Matériaux Construction': df['Materiaux_Construction_Tonnes'].mean(),
            'Véhicules': df['Vehicules_Tonnes'].mean(),
            'Produits Pétroliers': df['Produits_Petroliers_Tonnes'].mean(),
            'Ylang-Ylang': df['Ylang_Ylang_Tonnes'].mean(),
            'Vanille': df['Vanille_Tonnes'].mean()
        }
        
        total = sum(commodities.values())
        for commodity, volume in commodities.items():
            percentage = (volume / total) * 100
            print(f"  {commodity}: {percentage:.1f}%")
        
        # 4. Partenaires commerciaux
        print("\n4. 🌍 PRINCIPAUX PARTENAIRES:")
        partners = {
            'France': df['France_Import_Tonnes'].mean(),
            'Madagascar': df['Madagascar_Import_Tonnes'].mean(),
            'Comores': df['Comores_Import_Tonnes'].mean(),
            'Afrique': df['Afrique_Import_Tonnes'].mean()
        }
        
        total_import = df['Import_Tonnes'].mean()
        for partner, volume in partners.items():
            percentage = (volume / total_import) * 100
            print(f"  {partner}: {percentage:.1f}% des importations")
        
        # 5. Recommandations
        print("\n5. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        print("• Développer les exportations de produits de valeur (ylang-ylang, vanille)")
        print("• Renforcer les échanges commerciaux avec les pays voisins (Madagascar, Comores)")
        print("• Investir dans la transformation locale pour réduire les importations alimentaires")
        print("• Moderniser les infrastructures portuaires pour accroître la capacité")
        print("• Diversifier les partenaires commerciaux au-delà de la France")

def main():
    """Fonction principale"""
    print("🏝️  ANALYSE IMPORT-EXPORT - PORT DE LONGONI, MAYOTTE")
    print("=" * 60)
    
    # Initialiser l'analyseur
    analyzer = PortMayotteTradeAnalyzer()
    
    # Générer les données
    trade_data = analyzer.generate_trade_data()
    
    # Sauvegarder les données
    output_file = 'port_mayotte_trade_data.csv'
    trade_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(trade_data[['Date', 'Import_Tonnes', 'Export_Tonnes', 'Balance_Commerciale']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse commerciale...")
    analyzer.create_trade_analysis(trade_data)
    
    print(f"\n✅ Analyse du {analyzer.port_name} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Tonnage, valeur, partenaires, types de marchandises")

if __name__ == "__main__":
    main()