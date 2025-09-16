import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortPolynesieTradeAnalyzer:
    def __init__(self):
        self.port_name = "Port de Papeete - Polynésie Française"
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', 
                      '#AB83A1', '#5CAB7D', '#2A9D8F', '#E76F51', '#264653']
        
        self.start_year = 2010
        self.end_year = 2024
        
    def generate_trade_data(self):
        """Génère des données d'import-export pour le Port de Polynésie Française"""
        print("📊 Génération des données de commerce pour le Port de Polynésie...")
        
        # Créer une base de données mensuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='M')
        
        data = {'Date': dates}
        
        # Données d'importation (en tonnes)
        data['Import_Tonnes'] = self._simulate_imports(dates)
        data['Import_Valeur_EUR'] = data['Import_Tonnes'] * np.random.uniform(1000, 1500, len(dates))  # Valeur plus élevée due à l'éloignement
        
        # Données d'exportation (en tonnes)
        data['Export_Tonnes'] = self._simulate_exports(dates)
        data['Export_Valeur_EUR'] = data['Export_Tonnes'] * np.random.uniform(2000, 5000, len(dates))  # Produits de haute valeur
        
        # Balance commerciale
        data['Balance_Commerciale'] = data['Export_Valeur_EUR'] - data['Import_Valeur_EUR']
        
        # Principales catégories de marchandises
        data['Alimentaire_Tonnes'] = self._simulate_food_trade(dates)
        data['Materiaux_Tonnes'] = self._simulate_materials_trade(dates)
        data['Vehicules_Tonnes'] = self._simulate_vehicles_trade(dates)
        data['Produits_Lux_Tonnes'] = self._simulate_luxury_trade(dates)  # Spécifique à la Polynésie
        data['Perles_Tonnes'] = self._simulate_pearls_trade(dates)  # Exportation importante
        
        # Principaux partenaires commerciaux
        data['France_Import_Tonnes'] = self._simulate_france_trade(dates)
        data['Asie_Import_Tonnes'] = self._simulate_asia_trade(dates)
        data['USA_Import_Tonnes'] = self._simulate_usa_trade(dates)  # Partenaire important pour la Polynésie
        data['NZ_Import_Tonnes'] = self._simulate_nz_trade(dates)  # Nouvelle-Zélande
        
        # Activité portuaire
        data['Navires_Entrees'] = self._simulate_ship_arrivals(dates)
        data['Conteneurs_Traites'] = self._simulate_containers(dates)
        data['Touristes_Croisieres'] = self._simulate_cruise_tourists(dates)  # Important pour la Polynésie
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances saisonnières et économiques
        self._add_economic_trends(df)
        
        return df
    
    def _simulate_imports(self, dates):
        """Simule les importations du Port de Polynésie"""
        base_imports = 40000  # tonnes/mois de base (moins qu'à La Réunion)
        
        imports = []
        for i, date in enumerate(dates):
            # Croissance annuelle de 2-4%
            growth = 1 + (0.02 + 0.02 * (i / len(dates)))
            
            # Saisonnalité (pic avant la haute saison touristique)
            month = date.month
            if month in [6, 7, 8]:  # Pic saison touristique
                seasonal = 1.3
            elif month in [1, 2]:  # Saison des pluies
                seasonal = 0.8
            else:
                seasonal = 1.0
            
            # Bruit aléatoire
            noise = np.random.normal(1, 0.12)  # Plus volatile due à l'isolement
            
            imports.append(base_imports * growth * seasonal * noise)
        
        return imports
    
    def _simulate_exports(self, dates):
        """Simule les exportations du Port de Polynésie"""
        base_exports = 8000  # tonnes/mois de base (exportations limitées)
        
        exports = []
        for i, date in enumerate(dates):
            # Croissance modérée
            growth = 1 + (0.01 + 0.01 * (i / len(dates)))
            
            # Saisonnalité différente
            month = date.month
            if month in [9, 10, 11]:  # Pic production perlière
                seasonal = 1.5
            elif month in [1, 2]:  # Saison des pluies
                seasonal = 0.6
            else:
                seasonal = 1.0
            
            noise = np.random.normal(1, 0.2)  # Plus volatile
            
            exports.append(base_exports * growth * seasonal * noise)
        
        return exports
    
    def _simulate_food_trade(self, dates):
        """Simule le commerce alimentaire (produits importés, vanille, etc.)"""
        base_volume = 18000
        
        volumes = []
        for date in dates:
            month = date.month
            # Pic pendant la haute saison touristique
            if 6 <= month <= 8:
                seasonal = 1.5
            else:
                seasonal = 0.9
            
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * seasonal * noise)
        
        return volumes
    
    def _simulate_materials_trade(self, dates):
        """Simule les matériaux de construction"""
        base_volume = 12000
        
        volumes = []
        for date in dates:
            # Croissance régulière liée au développement touristique
            growth = 1 + 0.015 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.1)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_vehicles_trade(self, dates):
        """Simule l'importation de véhicules"""
        base_volume = 2000
        
        volumes = []
        for date in dates:
            # Moins de saisonnalité marquée
            growth = 1 + 0.03 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_luxury_trade(self, dates):
        """Simule les produits de luxe (spécifique à la Polynésie)"""
        base_volume = 1000
        
        volumes = []
        for date in dates:
            # Forte saisonnalité liée au tourisme
            month = date.month
            if 6 <= month <= 8:
                seasonal = 2.0
            else:
                seasonal = 0.7
            
            growth = 1 + 0.05 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.25)
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_pearls_trade(self, dates):
        """Simule l'exportation de perles de Tahiti"""
        base_volume = 500
        
        volumes = []
        for date in dates:
            # Saisonnalité de la production perlière
            month = date.month
            if 9 <= month <= 11:
                seasonal = 1.8
            else:
                seasonal = 0.6
            
            # Croissance irrégulière (marché de niche)
            growth = 1 + 0.02 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.3)  # Très volatile
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_france_trade(self, dates):
        """Simule le commerce avec la France métropolitaine"""
        base_volume = 20000
        
        volumes = []
        for date in dates:
            # Relation stable mais moins importante qu'à La Réunion
            growth = 1 + 0.01 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.1)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_asia_trade(self, dates):
        """Simule le commerce avec l'Asie"""
        base_volume = 8000
        
        volumes = []
        for date in dates:
            # Forte croissance du commerce asiatique (notamment pour les perles)
            growth = 1 + 0.07 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.2)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_usa_trade(self, dates):
        """Simule le commerce avec les USA"""
        base_volume = 6000
        
        volumes = []
        for date in dates:
            # Croissance modérée
            growth = 1 + 0.04 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_nz_trade(self, dates):
        """Simule le commerce avec la Nouvelle-Zélande"""
        base_volume = 3000
        
        volumes = []
        for date in dates:
            # Croissance régulière
            growth = 1 + 0.03 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.18)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_ship_arrivals(self, dates):
        """Simule le nombre de navires entrants"""
        base_arrivals = 80  # Moins qu'à La Réunion
        
        arrivals = []
        for date in dates:
            # Saisonnalité maritime
            month = date.month
            if month in [5, 6, 7, 8, 9]:  # Meilleures conditions météo
                seasonal = 1.2
            elif month in [1, 2]:  # Saison des pluies/cyclones
                seasonal = 0.6
            else:
                seasonal = 1.0
            
            growth = 1 + 0.01 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.08)
            arrivals.append(base_arrivals * growth * seasonal * noise)
        
        return arrivals
    
    def _simulate_containers(self, dates):
        """Simule le nombre de conteneurs traités"""
        base_containers = 15000  # Moins qu'à La Réunion
        
        containers = []
        for date in dates:
            growth = 1 + 0.04 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.1)
            containers.append(base_containers * growth * noise)
        
        return containers
    
    def _simulate_cruise_tourists(self, dates):
        """Simule le nombre de touristes de croisière"""
        base_tourists = 5000
        
        tourists = []
        for date in dates:
            # Forte saisonnalité
            month = date.month
            if 6 <= month <= 9:
                seasonal = 2.5
            elif month in [1, 2, 12]:
                seasonal = 0.4
            else:
                seasonal = 1.0
            
            # Croissance du tourisme de croisière
            growth = 1 + 0.06 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.25)
            tourists.append(base_tourists * growth * seasonal * noise)
        
        return tourists
    
    def _add_economic_trends(self, df):
        """Ajoute des tendances économiques réalistes"""
        for i, row in df.iterrows():
            date = row['Date']
            year = date.year
            
            # Impact COVID-19 (2020-2021) - très fort sur le tourisme
            if 2020 <= year <= 2021:
                if year == 2020 and date.month in [3, 4, 5]:
                    df.loc[i, 'Import_Tonnes'] *= 0.5
                    df.loc[i, 'Export_Tonnes'] *= 0.4
                    df.loc[i, 'Navires_Entrees'] *= 0.4
                    df.loc[i, 'Touristes_Croisieres'] *= 0.1  # Effondrement du tourisme
            
            # Reprise post-COVID progressive
            elif year >= 2022:
                recovery = 1 + 0.015 * (year - 2022)
                df.loc[i, 'Import_Tonnes'] *= recovery
                df.loc[i, 'Export_Tonnes'] *= recovery
                df.loc[i, 'Touristes_Croisieres'] *= (1 + 0.1 * (year - 2022))  # Reprise plus forte du tourisme
            
            # Développement du tourisme (2015-2019)
            elif 2015 <= year <= 2019:
                df.loc[i, 'Touristes_Croisieres'] *= 1.1
            
            # Augmentation du commerce asiatique (notamment pour les perles)
            if year >= 2015:
                asia_growth = 1 + 0.04 * (year - 2015)
                df.loc[i, 'Asie_Import_Tonnes'] *= asia_growth
                df.loc[i, 'Perles_Tonnes'] *= (1 + 0.03 * (year - 2015))
    
    def create_trade_analysis(self, df):
        """Crée une analyse complète du commerce portuaire"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 18))
        
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
        
        # 6. Tourisme de croisière
        ax6 = plt.subplot(3, 2, 6)
        self._plot_cruise_tourism(df, ax6)
        
        plt.suptitle(f'Analyse Import-Export - {self.port_name} (2010-2024)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('port_polynesie_trade_analysis.png', dpi=300, bbox_inches='tight')
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
        commodities = ['Alimentaire_Tonnes', 'Materiaux_Tonnes', 
                      'Vehicules_Tonnes', 'Produits_Lux_Tonnes', 'Perles_Tonnes']
        labels = ['Alimentaire', 'Matériaux', 'Véhicules', 'Produits Luxe', 'Perles']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572']
        
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
        partners = ['France_Import_Tonnes', 'Asie_Import_Tonnes', 'USA_Import_Tonnes', 'NZ_Import_Tonnes']
        labels = ['France', 'Asie', 'USA', 'Nouvelle-Zélande']
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
    
    def _plot_cruise_tourism(self, df, ax):
        """Plot du tourisme de croisière"""
        ax.plot(df['Date'], df['Touristes_Croisieres']/1000, 
               label='Touristes de Croisière', linewidth=2, color='#6A0572', alpha=0.8)
        
        ax.set_title('Tourisme de Croisière (milliers de touristes)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Touristes (x1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ajouter une tendance
        z = np.polyfit(range(len(df)), df['Touristes_Croisieres']/1000, 1)
        p = np.poly1d(z)
        ax.plot(df['Date'], p(range(len(df))), linestyle='--', color='black', 
               label='Tendance')
    
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
            'Alimentaire': df['Alimentaire_Tonnes'].mean(),
            'Matériaux': df['Materiaux_Tonnes'].mean(),
            'Véhicules': df['Vehicules_Tonnes'].mean(),
            'Produits Luxe': df['Produits_Lux_Tonnes'].mean(),
            'Perles': df['Perles_Tonnes'].mean()
        }
        
        total = sum(commodities.values())
        for commodity, volume in commodities.items():
            percentage = (volume / total) * 100
            print(f"  {commodity}: {percentage:.1f}%")
        
        # 4. Partenaires commerciaux
        print("\n4. 🌍 PRINCIPAUX PARTENAIRES:")
        partners = {
            'France': df['France_Import_Tonnes'].mean(),
            'Asie': df['Asie_Import_Tonnes'].mean(),
            'USA': df['USA_Import_Tonnes'].mean(),
            'Nouvelle-Zélande': df['NZ_Import_Tonnes'].mean()
        }
        
        total_import = df['Import_Tonnes'].mean()
        for partner, volume in partners.items():
            percentage = (volume / total_import) * 100
            print(f"  {partner}: {percentage:.1f}% des importations")
        
        # 5. Tourisme de croisière
        print("\n5. 🚢 TOURISME DE CROISIÈRE:")
        avg_tourists = df['Touristes_Croisieres'].mean()
        max_tourists = df['Touristes_Croisieres'].max()
        print(f"Touristes de croisière moyens/mois: {avg_tourists:.0f}")
        print(f"Pic de touristes de croisière: {max_tourists:.0f}")
        
        # 6. Recommandations
        print("\n6. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        print("• Développer les exportations de produits à haute valeur ajoutée (perles, vanille)")
        print("• Diversifier les partenaires commerciaux vers l'Asie et les USA")
        print("• Investir dans les infrastructures pour accueillir plus de navires de croisière")
        print("• Promouvoir les produits locaux (nonoi, monoï, vanille) à l'exportation")
        print("• Renforcer les liens commerciaux avec la Nouvelle-Zélande et l'Australie")

def main():
    """Fonction principale"""
    print("🏝️  ANALYSE IMPORT-EXPORT - PORT DE POLYNÉSIE FRANÇAISE")
    print("=" * 60)
    
    # Initialiser l'analyseur
    analyzer = PortPolynesieTradeAnalyzer()
    
    # Générer les données
    trade_data = analyzer.generate_trade_data()
    
    # Sauvegarder les données
    output_file = 'port_polynesie_trade_data.csv'
    trade_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(trade_data[['Date', 'Import_Tonnes', 'Export_Tonnes', 'Balance_Commerciale', 'Touristes_Croisieres']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse commerciale...")
    analyzer.create_trade_analysis(trade_data)
    
    print(f"\n✅ Analyse du {analyzer.port_name} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Tonnage, valeur, partenaires, types de marchandises, tourisme")

if __name__ == "__main__":
    main()