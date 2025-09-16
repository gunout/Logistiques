import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortWallisFutunaTradeAnalyzer:
    def __init__(self):
        self.port_name = "Port de Mata'Utu - Wallis-et-Futuna"
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', 
                      '#AB83A1', '#5CAB7D', '#2A9D8F', '#E76F51', '#264653']
        
        self.start_year = 2010
        self.end_year = 2024
        
    def generate_trade_data(self):
        """Génère des données d'import-export pour le Port de Wallis-et-Futuna"""
        print("📊 Génération des données de commerce pour Wallis-et-Futuna...")
        
        # Créer une base de données mensuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='M')
        
        data = {'Date': dates}
        
        # Données d'importation (en tonnes)
        data['Import_Tonnes'] = self._simulate_imports(dates)
        data['Import_Valeur_EUR'] = data['Import_Tonnes'] * np.random.uniform(1500, 2500, len(dates))  # Valeur élevée due à l'isolement extrême
        
        # Données d'exportation (en tonnes)
        data['Export_Tonnes'] = self._simulate_exports(dates)
        data['Export_Valeur_EUR'] = data['Export_Tonnes'] * np.random.uniform(1000, 3000, len(dates))  # Produits de niche
        
        # Balance commerciale
        data['Balance_Commerciale'] = data['Export_Valeur_EUR'] - data['Import_Valeur_EUR']
        
        # Principales catégories de marchandises
        data['Alimentaire_Tonnes'] = self._simulate_food_trade(dates)
        data['Materiaux_Tonnes'] = self._simulate_materials_trade(dates)
        data['Vehicules_Tonnes'] = self._simulate_vehicles_trade(dates)
        data['Produits_Petroliers_Tonnes'] = self._simulate_petroleum_trade(dates)
        data['Coprah_Tonnes'] = self._simulate_coprah_trade(dates)  # Exportation traditionnelle
        data['Artisanat_Tonnes'] = self._simulate_handicraft_trade(dates)  # Produit local
        
        # Principaux partenaires commerciaux
        data['France_Import_Tonnes'] = self._simulate_france_trade(dates)
        data['Nouvelle_Caledonie_Import_Tonnes'] = self._simulate_nc_trade(dates)  # Partenaire régional important
        data['Australie_Import_Tonnes'] = self._simulate_australia_trade(dates)
        data['Fidji_Import_Tonnes'] = self._simulate_fiji_trade(dates)  # Partenaire régional
        
        # Activité portuaire
        data['Navires_Entrees'] = self._simulate_ship_arrivals(dates)
        data['Conteneurs_Traites'] = self._simulate_containers(dates)
        data['Cargaisons_Speciales'] = self._simulate_special_cargo(dates)  # Approvisionnements spéciaux
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances saisonnières et économiques
        self._add_economic_trends(df)
        
        return df
    
    def _simulate_imports(self, dates):
        """Simule les importations du Port de Mata'Utu"""
        base_imports = 8000  # tonnes/mois de base (très faible volume)
        
        imports = []
        for i, date in enumerate(dates):
            # Croissance annuelle très modeste
            growth = 1 + (0.005 + 0.005 * (i / len(dates)))
            
            # Saisonnalité (pic avant la saison des pluies)
            month = date.month
            if month in [9, 10, 11]:  # Préparation pour la saison des pluies
                seasonal = 1.5
            elif month in [1, 2, 3]:  # Saison des cyclones
                seasonal = 0.4
            else:
                seasonal = 1.0
            
            # Bruit aléatoire
            noise = np.random.normal(1, 0.2)  # Très volatile due à l'isolement extrême
            
            imports.append(base_imports * growth * seasonal * noise)
        
        return imports
    
    def _simulate_exports(self, dates):
        """Simule les exportations du Port de Mata'Utu"""
        base_exports = 2000  # tonnes/mois de base (très faible)
        
        exports = []
        for i, date in enumerate(dates):
            # Croissance très modeste
            growth = 1 + (0.002 + 0.003 * (i / len(dates)))
            
            # Saisonnalité liée à la production de coprah
            month = date.month
            if month in [6, 7, 8]:  # Meilleure saison pour le coprah
                seasonal = 2.0
            elif month in [1, 2, 12]:  # Saison des pluies/cyclones
                seasonal = 0.3
            else:
                seasonal = 0.8
            
            noise = np.random.normal(1, 0.3)  # Extrêmement volatile
            
            exports.append(base_exports * growth * seasonal * noise)
        
        return exports
    
    def _simulate_food_trade(self, dates):
        """Simule le commerce alimentaire"""
        base_volume = 4000
        
        volumes = []
        for date in dates:
            month = date.month
            # Pic avant la saison des pluies
            if 9 <= month <= 11:
                seasonal = 1.7
            else:
                seasonal = 0.9
            
            noise = np.random.normal(1, 0.25)
            volumes.append(base_volume * seasonal * noise)
        
        return volumes
    
    def _simulate_materials_trade(self, dates):
        """Simule les matériaux de construction"""
        base_volume = 2000
        
        volumes = []
        for date in dates:
            # Saisonnalité construction (saison sèche)
            month = date.month
            if 5 <= month <= 10:
                seasonal = 1.6
            else:
                seasonal = 0.6
            
            # Croissance très modeste
            growth = 1 + 0.003 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.2)
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_vehicles_trade(self, dates):
        """Simule l'importation de véhicules"""
        base_volume = 200
        
        volumes = []
        for date in dates:
            # Très peu de saisonnalité
            growth = 1 + 0.005 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.4)  # Extrêmement volatile (très petits volumes)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_petroleum_trade(self, dates):
        """Simule les produits pétroliers"""
        base_volume = 1500
        
        volumes = []
        for date in dates:
            # Pic avant la saison des pluies
            month = date.month
            if 9 <= month <= 11:
                seasonal = 1.8
            else:
                seasonal = 0.9
            
            # Croissance stable
            growth = 1 + 0.005 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_coprah_trade(self, dates):
        """Simule l'exportation de coprah (noix de coco séchée)"""
        base_volume = 800
        
        volumes = []
        for date in dates:
            # Forte saisonnalité
            month = date.month
            if 6 <= month <= 8:
                seasonal = 2.5
            elif month in [1, 2, 12]:
                seasonal = 0.2
            else:
                seasonal = 0.7
            
            # Déclin progressif de cette activité traditionnelle
            year_factor = 1 - 0.01 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.4)
            volumes.append(base_volume * year_factor * seasonal * noise)
        
        return volumes
    
    def _simulate_handicraft_trade(self, dates):
        """Simule l'exportation d'artisanat local"""
        base_volume = 100
        
        volumes = []
        for date in dates:
            # Pic pendant la saison touristique (très limitée)
            month = date.month
            if 7 <= month <= 9:
                seasonal = 1.8
            else:
                seasonal = 0.7
            
            # Légère croissance
            growth = 1 + 0.01 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.5)  # Très volatile
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_france_trade(self, dates):
        """Simule le commerce avec la France métropolitaine"""
        base_volume = 5000
        
        volumes = []
        for date in dates:
            # Relation stable mais limitée
            growth = 1 + 0.003 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_nc_trade(self, dates):
        """Simule le commerce avec la Nouvelle-Calédonie (partenaire régional majeur)"""
        base_volume = 2000
        
        volumes = []
        for date in dates:
            # Croissance modérée
            growth = 1 + 0.008 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.2)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_australia_trade(self, dates):
        """Simule le commerce avec l'Australie"""
        base_volume = 500
        
        volumes = []
        for date in dates:
            # Croissance irrégulière
            growth = 1 + 0.005 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.3)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_fiji_trade(self, dates):
        """Simule le commerce avec Fidji"""
        base_volume = 300
        
        volumes = []
        for date in dates:
            # Croissance modeste
            growth = 1 + 0.004 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.35)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_ship_arrivals(self, dates):
        """Simule le nombre de navires entrants"""
        base_arrivals = 20  # Très faible volume
        
        arrivals = []
        for date in dates:
            # Saisonnalité maritime
            month = date.month
            if 5 <= month <= 10:  # Meilleures conditions météo
                seasonal = 1.4
            elif month in [1, 2, 3]:  # Saison des cyclones
                seasonal = 0.3
            else:
                seasonal = 1.0
            
            growth = 1 + 0.002 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            arrivals.append(base_arrivals * growth * seasonal * noise)
        
        return arrivals
    
    def _simulate_containers(self, dates):
        """Simule le nombre de conteneurs traités"""
        base_containers = 2000  # Très faible volume
        
        containers = []
        for date in dates:
            growth = 1 + 0.005 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.18)
            containers.append(base_containers * growth * noise)
        
        return containers
    
    def _simulate_special_cargo(self, dates):
        """Simule les approvisionnements spéciaux (matériel médical, éducatif, etc.)"""
        base_cargo = 300
        
        cargo = []
        for date in dates:
            # Peu de saisonnalité
            growth = 1 + 0.01 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.25)
            cargo.append(base_cargo * growth * noise)
        
        return cargo
    
    def _add_economic_trends(self, df):
        """Ajoute des tendances économiques réalistes"""
        for i, row in df.iterrows():
            date = row['Date']
            year = date.year
            
            # Impact COVID-19 (2020-2021) - très fort sur les micro-économies insulaires
            if 2020 <= year <= 2021:
                if year == 2020 and date.month in [3, 4, 5]:
                    df.loc[i, 'Import_Tonnes'] *= 0.3
                    df.loc[i, 'Export_Tonnes'] *= 0.2
                    df.loc[i, 'Navires_Entrees'] *= 0.4
                    df.loc[i, 'Artisanat_Tonnes'] *= 0.1  # Effondrement du tourisme
            
            # Reprise post-COVID très lente
            elif year >= 2022:
                recovery = 1 + 0.005 * (year - 2022)
                df.loc[i, 'Import_Tonnes'] *= recovery
                df.loc[i, 'Export_Tonnes'] *= recovery
            
            # Déclin continu du coprah
            if year >= 2015:
                coprah_decline = 1 - 0.015 * (year - 2015)
                df.loc[i, 'Coprah_Tonnes'] *= coprah_decline
            
            # Augmentation des échanges avec la Nouvelle-Calédonie
            if year >= 2018:
                nc_growth = 1 + 0.01 * (year - 2018)
                df.loc[i, 'Nouvelle_Caledonie_Import_Tonnes'] *= nc_growth
            
            # Aide au développement (augmentation des cargaisons spéciales)
            if year >= 2020:
                aid_growth = 1 + 0.02 * (year - 2020)
                df.loc[i, 'Cargaisons_Speciales'] *= aid_growth
    
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
        
        # 6. Produits locaux
        ax6 = plt.subplot(3, 2, 6)
        self._plot_local_products(df, ax6)
        
        plt.suptitle(f'Analyse Import-Export - {self.port_name} (2010-2024)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('port_wallis_futuna_trade_analysis.png', dpi=300, bbox_inches='tight')
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
                      'Vehicules_Tonnes', 'Produits_Petroliers_Tonnes']
        labels = ['Alimentaire', 'Matériaux', 'Véhicules', 'Pétrole']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602']
        
        for i, col in enumerate(commodities):
            ax.plot(df['Date'], df[col]/1000, label=labels[i], 
                   linewidth=2, color=colors[i], alpha=0.8)
        
        ax.set_title('Types de Marchandises Importées (milliers de tonnes)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Tonnes (x1000)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_trade_partners(self, df, ax):
        """Plot des partenaires commerciaux"""
        partners = ['France_Import_Tonnes', 'Nouvelle_Caledonie_Import_Tonnes', 
                   'Australie_Import_Tonnes', 'Fidji_Import_Tonnes']
        labels = ['France', 'Nouvelle-Calédonie', 'Australie', 'Fidji']
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
    
    def _plot_local_products(self, df, ax):
        """Plot des produits locaux d'exportation"""
        ax.plot(df['Date'], df['Coprah_Tonnes'], 
               label='Coprah Exporté', linewidth=2, color='#2A9D8F', alpha=0.8)
        
        ax2 = ax.twinx()
        ax2.plot(df['Date'], df['Artisanat_Tonnes']*10,  # Multiplier pour meilleure visualisation
                label='Artisanat Exporté (x10)', linewidth=2, color='#E76F51', alpha=0.8)
        
        ax.set_title('Produits Locaux d\'Exportation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coprah (tonnes)', color='#2A9D8F')
        ax2.set_ylabel('Artisanat (tonnes x10)', color='#E76F51')
        ax.tick_params(axis='y', labelcolor='#2A9D8F')
        ax2.tick_params(axis='y', labelcolor='#E76F51')
        ax.grid(True, alpha=0.3)
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _generate_trade_insights(self, df):
        """Génère des insights analytiques"""
        print(f"📊 INSIGHTS ANALYTIQUES - {self.port_name}")
        print("=" * 60)
        
        # 1. Statistiques de base
        print("\n1. 📈 STATISTIQUES GÉNÉRALES (2010-2024):")
        total_imports = df['Import_Tonnes'].sum() / 1e6
        total_exports = df['Export_Tonnes'].sum() / 1e6
        trade_balance = df['Balance_Commerciale'].mean() / 1e6
        
        print(f"Importations totales: {total_imports:.2f} millions de tonnes")
        print(f"Exportations totales: {total_exports:.2f} millions de tonnes")
        print(f"Balance commerciale moyenne: {trade_balance:.2f} M€/mois")
        
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
            'Pétrole': df['Produits_Petroliers_Tonnes'].mean()
        }
        
        total = sum(commodities.values())
        for commodity, volume in commodities.items():
            percentage = (volume / total) * 100
            print(f"  {commodity}: {percentage:.1f}%")
        
        # 4. Partenaires commerciaux
        print("\n4. 🌍 PRINCIPAUX PARTENAIRES:")
        partners = {
            'France': df['France_Import_Tonnes'].mean(),
            'Nouvelle-Calédonie': df['Nouvelle_Caledonie_Import_Tonnes'].mean(),
            'Australie': df['Australie_Import_Tonnes'].mean(),
            'Fidji': df['Fidji_Import_Tonnes'].mean()
        }
        
        total_import = df['Import_Tonnes'].mean()
        for partner, volume in partners.items():
            percentage = (volume / total_import) * 100
            print(f"  {partner}: {percentage:.1f}% des importations")
        
        # 5. Produits locaux
        print("\n5. 🥥 PRODUITS LOCAUX:")
        avg_coprah = df['Coprah_Tonnes'].mean()
        avg_handicraft = df['Artisanat_Tonnes'].mean()
        print(f"Coprah exporté moyen/mois: {avg_coprah:.0f} tonnes")
        print(f"Artisanat exporté moyen/mois: {avg_handicraft:.0f} tonnes")
        
        # 6. Recommandations
        print("\n6. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        print("• Développer des produits d'exportation à plus haute valeur ajoutée")
        print("• Renforcer les liens commerciaux avec les partenaires régionaux (Nouvelle-Calédonie, Fidji)")
        print("• Investir dans la transformation locale des produits (huile de coco, etc.)")
        print("• Développer un tourisme durable et respectueux de la culture locale")
        print("• Améliorer les infrastructures portuaires pour réduire les coûts logistiques")
        print("• Diversifier l'économie au-delà de la dépendance aux importations")

def main():
    """Fonction principale"""
    print("🏝️  ANALYSE IMPORT-EXPORT - PORT DE WALLIS-ET-FUTUNA")
    print("=" * 60)
    
    # Initialiser l'analyseur
    analyzer = PortWallisFutunaTradeAnalyzer()
    
    # Générer les données
    trade_data = analyzer.generate_trade_data()
    
    # Sauvegarder les données
    output_file = 'port_wallis_futuna_trade_data.csv'
    trade_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(trade_data[['Date', 'Import_Tonnes', 'Export_Tonnes', 'Balance_Commerciale', 'Navires_Entrees']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse commerciale...")
    analyzer.create_trade_analysis(trade_data)
    
    print(f"\n✅ Analyse du {analyzer.port_name} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Tonnage, valeur, partenaires, types de marchandises, produits locaux")

if __name__ == "__main__":
    main()