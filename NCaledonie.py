import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortNoumeaTradeAnalyzer:
    def __init__(self):
        self.port_name = "Port de Nouméa - Nouvelle-Calédonie"
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', 
                      '#AB83A1', '#5CAB7D', '#2A9D8F', '#E76F51', '#264653']
        
        self.start_year = 2010
        self.end_year = 2024
        
    def generate_trade_data(self):
        """Génère des données d'import-export pour le Port de Nouméa"""
        print("📊 Génération des données de commerce pour le Port de Nouméa...")
        
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='M')
        
        data = {'Date': dates}
        
        # Données d'importation
        data['Import_Tonnes'] = self._simulate_imports(dates)
        data['Import_Valeur_EUR'] = data['Import_Tonnes'] * np.random.uniform(1000, 1500, len(dates))
        
        # Données d'exportation (dominée par le nickel)
        data['Export_Tonnes'] = self._simulate_exports(dates)
        data['Export_Valeur_EUR'] = data['Export_Tonnes'] * np.random.uniform(800, 1200, len(dates))
        
        # Balance commerciale
        data['Balance_Commerciale'] = data['Export_Valeur_EUR'] - data['Import_Valeur_EUR']
        
        # Catégories de marchandises spécifiques à la NC
        data['Nickel_Tonnes'] = self._simulate_nickel_exports(dates)
        data['Minerais_Tonnes'] = self._simulate_minerals_trade(dates)
        data['Produits_Alimentaires_Tonnes'] = self._simulate_food_trade(dates)
        data['Materiaux_Construction_Tonnes'] = self._simulate_construction_materials(dates)
        data['Vehicules_Tonnes'] = self._simulate_vehicles_trade(dates)
        data['Produits_Chimiques_Tonnes'] = self._simulate_chemicals_trade(dates)
        
        # Partenaires commerciaux
        data['France_Import_Tonnes'] = self._simulate_france_trade(dates)
        data['Australie_Import_Tonnes'] = self._simulate_australia_trade(dates)
        data['Asie_Import_Tonnes'] = self._simulate_asia_trade(dates)
        data['Afrique_Import_Tonnes'] = self._simulate_africa_trade(dates)
        
        # Activité portuaire
        data['Navires_Entrees'] = self._simulate_ship_arrivals(dates)
        data['Conteneurs_Traites'] = self._simulate_containers(dates)
        
        df = pd.DataFrame(data)
        self._add_economic_trends(df)
        
        return df

    def _simulate_imports(self, dates):
        """Simule les importations du Port de Nouméa"""
        base_imports = 45000
        imports = []
        for i, date in enumerate(dates):
            growth = 1 + (0.04 + 0.02 * (i / len(dates)))
            month = date.month
            if month in [1, 2]:  # Saison cyclonique
                seasonal = 0.7
            elif month in [9, 10]:  # Pic pré-fêtes
                seasonal = 1.2
            else:
                seasonal = 1.0
            noise = np.random.normal(1, 0.12)
            imports.append(base_imports * growth * seasonal * noise)
        return imports

    def _simulate_exports(self, dates):
        """Simule les exportations (forte dépendance au nickel)"""
        base_exports = 35000
        exports = []
        for i, date in enumerate(dates):
            growth = 1 + (0.03 + 0.01 * (i / len(dates)))
            # Saisonnalité liée à la production minière
            if date.month in [6, 7, 8]:  # Pic production nickel
                seasonal = 1.4
            elif date.month in [1, 2]:  # Saison cyclonique
                seasonal = 0.6
            else:
                seasonal = 1.0
            noise = np.random.normal(1, 0.2)  # Forte volatilité
            exports.append(base_exports * growth * seasonal * noise)
        return exports

    def _simulate_nickel_exports(self, dates):
        """Simule les exportations de nickel"""
        base_nickel = 25000
        nickel = []
        for date in dates:
            # Forte dépendance aux cours du nickel
            price_effect = np.random.uniform(0.8, 1.3)
            if date.month in [6, 7, 8]:
                seasonal = 1.3
            else:
                seasonal = 0.9
            noise = np.random.normal(1, 0.25)
            nickel.append(base_nickel * price_effect * seasonal * noise)
        return nickel

    def _simulate_minerals_trade(self, dates):
        """Simule le commerce des autres minerais"""
        base_volume = 8000
        volumes = []
        for date in dates:
            growth = 1 + 0.02 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * noise)
        return volumes

    def _simulate_food_trade(self, dates):
        """Simule les importations alimentaires"""
        base_volume = 12000
        volumes = []
        for date in dates:
            if date.month in [11, 12]:  # Pic festif
                seasonal = 1.3
            else:
                seasonal = 1.0
            noise = np.random.normal(1, 0.1)
            volumes.append(base_volume * seasonal * noise)
        return volumes

    def _simulate_construction_materials(self, dates):
        """Simule les matériaux de construction"""
        base_volume = 10000
        volumes = []
        for date in dates:
            growth = 1 + 0.05 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.1)
            volumes.append(base_volume * growth * noise)
        return volumes

    def _simulate_vehicles_trade(self, dates):
        """Simule l'importation de véhicules"""
        base_volume = 3000
        volumes = []
        for date in dates:
            # Pic en début d'année (nouveaux modèles)
            if date.month in [1, 2]:
                seasonal = 1.5
            else:
                seasonal = 1.0
            
            growth = 1 + 0.04 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.12)
            volumes.append(base_volume * growth * seasonal * noise)
        return volumes

    def _simulate_chemicals_trade(self, dates):
        """Simule les produits chimiques et pharmaceutiques"""
        base_volume = 5000
        volumes = []
        for date in dates:
            growth = 1 + 0.03 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.06)
            volumes.append(base_volume * growth * noise)
        return volumes

    def _simulate_france_trade(self, dates):
        """Simule le commerce avec la France métropolitaine"""
        base_volume = 20000
        volumes = []
        for date in dates:
            growth = 1 + 0.02 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.07)
            volumes.append(base_volume * growth * noise)
        return volumes

    def _simulate_australia_trade(self, dates):
        """Commerce avec l'Australie (voisin important)"""
        base_volume = 15000
        volumes = []
        for date in dates:
            growth = 1 + 0.06 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.1)
            volumes.append(base_volume * growth * noise)
        return volumes

    def _simulate_asia_trade(self, dates):
        """Simule le commerce avec l'Asie"""
        base_volume = 10000
        volumes = []
        for date in dates:
            # Forte croissance du commerce asiatique
            growth = 1 + 0.08 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * noise)
        return volumes

    def _simulate_africa_trade(self, dates):
        """Simule le commerce avec l'Afrique"""
        base_volume = 3000
        volumes = []
        for date in dates:
            # Croissance modérée
            growth = 1 + 0.04 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.2)  # Plus volatile
            volumes.append(base_volume * growth * noise)
        return volumes

    def _simulate_ship_arrivals(self, dates):
        """Simule le nombre de navires entrants"""
        base_arrivals = 100
        
        arrivals = []
        for date in dates:
            # Saisonnalité maritime
            month = date.month
            if month in [5, 6, 9, 10]:  # Meilleures conditions météo
                seasonal = 1.1
            elif month in [1, 2]:  # Saison cyclonique
                seasonal = 0.7
            else:
                seasonal = 1.0
            
            growth = 1 + 0.015 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.05)
            arrivals.append(base_arrivals * growth * seasonal * noise)
        
        return arrivals

    def _simulate_containers(self, dates):
        """Simule le nombre de conteneurs traités"""
        base_containers = 20000
        
        containers = []
        for date in dates:
            growth = 1 + 0.05 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.08)
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
                    df.loc[i, 'Import_Tonnes'] *= 0.6
                    df.loc[i, 'Export_Tonnes'] *= 0.5
                    df.loc[i, 'Navires_Entrees'] *= 0.7
            
            # Croissance économique post-COVID
            elif year >= 2022:
                recovery = 1 + 0.02 * (year - 2022)
                df.loc[i, 'Import_Tonnes'] *= recovery
                df.loc[i, 'Export_Tonnes'] *= recovery
            
            # Développement des infrastructures (2015-2018)
            elif 2015 <= year <= 2018:
                df.loc[i, 'Materiaux_Construction_Tonnes'] *= 1.1
            
            # Augmentation du commerce asiatique
            if year >= 2018:
                asia_growth = 1 + 0.03 * (year - 2018)
                df.loc[i, 'Asie_Import_Tonnes'] *= asia_growth

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
        plt.savefig('port_noumea_trade_analysis.png', dpi=300, bbox_inches='tight')
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
        commodities = ['Nickel_Tonnes', 'Minerais_Tonnes', 
                      'Produits_Alimentaires_Tonnes', 'Materiaux_Construction_Tonnes']
        labels = ['Nickel', 'Autres Minerais', 'Alimentaire', 'Matériaux Construction']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602']
        
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
        partners = ['France_Import_Tonnes', 'Australie_Import_Tonnes', 'Asie_Import_Tonnes']
        labels = ['France', 'Australie', 'Asie']
        colors = ['#264653', '#2A9D8F', '#E76F51']
        
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
            'Nickel': df['Nickel_Tonnes'].mean(),
            'Autres Minerais': df['Minerais_Tonnes'].mean(),
            'Alimentaire': df['Produits_Alimentaires_Tonnes'].mean(),
            'Matériaux Construction': df['Materiaux_Construction_Tonnes'].mean()
        }
        
        total = sum(commodities.values())
        for commodity, volume in commodities.items():
            percentage = (volume / total) * 100
            print(f"  {commodity}: {percentage:.1f}%")
        
        # 4. Partenaires commerciaux
        print("\n4. 🌍 PRINCIPAUX PARTENAIRES:")
        partners = {
            'France': df['France_Import_Tonnes'].mean(),
            'Australie': df['Australie_Import_Tonnes'].mean(),
            'Asie': df['Asie_Import_Tonnes'].mean()
        }
        
        total_import = df['Import_Tonnes'].mean()
        for partner, volume in partners.items():
            percentage = (volume / total_import) * 100
            print(f"  {partner}: {percentage:.1f}% des importations")
        
        # 5. Recommandations
        print("\n5. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        print("• Diversifier l'économie au-delà du nickel pour réduire la dépendance")
        print("• Renforcer les liens commerciaux avec l'Australie et l'Asie")
        print("• Développer la transformation locale des matières premières")
        print("• Investir dans les infrastructures portuaires pour augmenter la capacité")
        print("• Promouvoir les produits locaux à l'exportation")

def main():
    """Fonction principale"""
    print("🏝️  ANALYSE IMPORT-EXPORT - PORT DE NOUMÉA")
    print("=" * 60)
    
    # Initialiser l'analyseur
    analyzer = PortNoumeaTradeAnalyzer()
    
    # Générer les données
    trade_data = analyzer.generate_trade_data()
    
    # Sauvegarder les données
    output_file = 'port_noumea_trade_data.csv'
    trade_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(trade_data[['Date', 'Import_Tonnes', 'Export_Tonnes', 'Nickel_Tonnes']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse commerciale...")
    analyzer.create_trade_analysis(trade_data)
    
    print(f"\n✅ Analyse du {analyzer.port_name} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Tonnage, valeur, partenaires, types de marchandises")

if __name__ == "__main__":
    main()