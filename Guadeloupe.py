import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortGuadeloupeTradeAnalyzer:
    def __init__(self):
        self.port_name = "Port de la Guadeloupe"
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', 
                      '#AB83A1', '#5CAB7D', '#2A9D8F', '#E76F51', '#264653']
        
        self.start_year = 2010
        self.end_year = 2024
        
    def generate_trade_data(self):
        """Génère des données d'import-export pour le Port de la Guadeloupe"""
        print("📊 Génération des données de commerce pour le Port de la Guadeloupe...")
        
        # Créer une base de données mensuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='M')
        
        data = {'Date': dates}
        
        # Données d'importation (en tonnes)
        data['Import_Tonnes'] = self._simulate_imports(dates)
        data['Import_Valeur_EUR'] = data['Import_Tonnes'] * np.random.uniform(900, 1300, len(dates))
        
        # Données d'exportation (en tonnes)
        data['Export_Tonnes'] = self._simulate_exports(dates)
        data['Export_Valeur_EUR'] = data['Export_Tonnes'] * np.random.uniform(700, 1000, len(dates))
        
        # Balance commerciale
        data['Balance_Commerciale'] = data['Export_Valeur_EUR'] - data['Import_Valeur_EUR']
        
        # Principales catégories de marchandises (spécifiques à la Guadeloupe)
        data['Alimentaire_Tonnes'] = self._simulate_food_trade(dates)
        data['Materiaux_Tonnes'] = self._simulate_materials_trade(dates)
        data['Produits_Petroliers_Tonnes'] = self._simulate_petroleum_trade(dates)
        data['Vehicules_Tonnes'] = self._simulate_vehicles_trade(dates)
        data['Sucre_Rhum_Tonnes'] = self._simulate_sugar_rum_trade(dates)
        data['Bananes_Tonnes'] = self._simulate_banana_trade(dates)
        
        # Principaux partenaires commerciaux
        data['France_Import_Tonnes'] = self._simulate_france_trade(dates)
        data['Caraibes_Import_Tonnes'] = self._simulate_caribbean_trade(dates)
        data['Etats_Unis_Import_Tonnes'] = self._simulate_usa_trade(dates)
        data['Europe_Import_Tonnes'] = self._simulate_europe_trade(dates)
        
        # Activité portuaire
        data['Navires_Entrees'] = self._simulate_ship_arrivals(dates)
        data['Conteneurs_Traites'] = self._simulate_containers(dates)
        data['Passagers_Croisiere'] = self._simulate_cruise_passengers(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances saisonnières et économiques
        self._add_economic_trends(df)
        
        return df
    
    def _simulate_imports(self, dates):
        """Simule les importations du Port de la Guadeloupe"""
        base_imports = 45000  # tonnes/mois de base
        
        imports = []
        for i, date in enumerate(dates):
            # Croissance annuelle de 2-4%
            growth = 1 + (0.02 + 0.02 * (i / len(dates)))
            
            # Saisonnalité (pic pendant la haute saison touristique)
            month = date.month
            if month in [12, 1, 2]:  # Haute saison touristique
                seasonal = 1.2
            elif month in [9, 10]:  # Saison cyclonique
                seasonal = 0.85
            else:
                seasonal = 1.0
            
            # Bruit aléatoire
            noise = np.random.normal(1, 0.1)
            
            imports.append(base_imports * growth * seasonal * noise)
        
        return imports
    
    def _simulate_exports(self, dates):
        """Simule les exportations du Port de la Guadeloupe"""
        base_exports = 18000  # tonnes/mois de base
        
        exports = []
        for i, date in enumerate(dates):
            # Croissance plus lente que les imports
            growth = 1 + (0.015 + 0.01 * (i / len(dates)))
            
            # Saisonnalité différente
            month = date.month
            if month in [3, 4, 5]:  # Pic après récoltes
                seasonal = 1.3
            elif month in [9, 10]:  # Saison cyclonique
                seasonal = 0.7
            else:
                seasonal = 1.0
            
            noise = np.random.normal(1, 0.15)
            
            exports.append(base_exports * growth * seasonal * noise)
        
        return exports
    
    def _simulate_food_trade(self, dates):
        """Simule le commerce alimentaire"""
        base_volume = 20000
        
        volumes = []
        for date in dates:
            month = date.month
            # Pic pendant la haute saison touristique
            if 12 <= month <= 2:
                seasonal = 1.25
            else:
                seasonal = 0.9
            
            noise = np.random.normal(1, 0.12)
            volumes.append(base_volume * seasonal * noise)
        
        return volumes
    
    def _simulate_materials_trade(self, dates):
        """Simule les matériaux de construction"""
        base_volume = 15000
        
        volumes = []
        for date in dates:
            # Moins de saisonnalité mais croissance régulière
            growth = 1 + 0.018 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.09)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_petroleum_trade(self, dates):
        """Simule les produits pétroliers"""
        base_volume = 8000
        
        volumes = []
        for date in dates:
            # Croissance modérée
            growth = 1 + 0.012 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.07)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_vehicles_trade(self, dates):
        """Simule l'importation de véhicules"""
        base_volume = 3500
        
        volumes = []
        for date in dates:
            # Pic en début d'année (nouveaux modèles)
            if date.month in [1, 2]:
                seasonal = 1.4
            else:
                seasonal = 1.0
            
            growth = 1 + 0.03 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_sugar_rum_trade(self, dates):
        """Simule l'exportation de sucre et rhum (spécifique à la Guadeloupe)"""
        base_volume = 6000
        
        volumes = []
        for date in dates:
            # Saisonnalité selon les récoltes de canne
            month = date.month
            if 2 <= month <= 6:  # Période de récolte et transformation
                seasonal = 1.4
            else:
                seasonal = 0.8
                
            growth = 1 + 0.02 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.18)
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_banana_trade(self, dates):
        """Simule l'exportation de bananes (spécifique à la Guadeloupe)"""
        base_volume = 7000
        
        volumes = []
        for date in dates:
            # Exportations relativement stables toute l'année
            month = date.month
            if month in [9, 10]:  # Impact saison cyclonique
                seasonal = 0.7
            else:
                seasonal = 1.0
                
            growth = 1 + 0.01 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * seasonal * noise)
        
        return volumes
    
    def _simulate_france_trade(self, dates):
        """Simule le commerce avec la France métropolitaine"""
        base_volume = 30000
        
        volumes = []
        for date in dates:
            # Relation stable avec la métropole
            growth = 1 + 0.02 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.08)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_caribbean_trade(self, dates):
        """Simule le commerce avec les Caraïbes"""
        base_volume = 5000
        
        volumes = []
        for date in dates:
            # Croissance modérée
            growth = 1 + 0.035 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.15)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_usa_trade(self, dates):
        """Simule le commerce avec les États-Unis"""
        base_volume = 4000
        
        volumes = []
        for date in dates:
            # Croissance régulière
            growth = 1 + 0.04 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.12)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_europe_trade(self, dates):
        """Simule le commerce avec l'Europe (hors France)"""
        base_volume = 6000
        
        volumes = []
        for date in dates:
            # Croissance modérée
            growth = 1 + 0.025 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.1)
            volumes.append(base_volume * growth * noise)
        
        return volumes
    
    def _simulate_ship_arrivals(self, dates):
        """Simule le nombre de navires entrants"""
        base_arrivals = 100
        
        arrivals = []
        for date in dates:
            # Saisonnalité maritime
            month = date.month
            if month in [12, 1, 2]:  # Haute saison croisières
                seasonal = 1.25
            elif month in [9, 10]:  # Saison cyclonique
                seasonal = 0.75
            else:
                seasonal = 1.0
            
            growth = 1 + 0.015 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.06)
            arrivals.append(base_arrivals * growth * seasonal * noise)
        
        return arrivals
    
    def _simulate_containers(self, dates):
        """Simule le nombre de conteneurs traités"""
        base_containers = 22000
        
        containers = []
        for date in dates:
            growth = 1 + 0.045 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.09)
            containers.append(base_containers * growth * noise)
        
        return containers
    
    def _simulate_cruise_passengers(self, dates):
        """Simule le nombre de passagers de croisière"""
        base_passengers = 40000
        
        passengers = []
        for date in dates:
            # Forte saisonnalité pour les croisières
            month = date.month
            if month in [12, 1, 2, 3]:  # Haute saison
                seasonal = 1.8
            elif month in [7, 8]:  # Saison estivale
                seasonal = 1.3
            elif month in [9, 10]:  # Saison cyclonique
                seasonal = 0.4
            else:
                seasonal = 0.9
                
            growth = 1 + 0.06 * ((date.year - self.start_year) / (self.end_year - self.start_year))
            noise = np.random.normal(1, 0.2)
            passengers.append(base_passengers * growth * seasonal * noise)
        
        return passengers
    
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
                    df.loc[i, 'Navires_Entrees'] *= 0.65
                    df.loc[i, 'Passagers_Croisiere'] *= 0.1  # Effondrement du tourisme
            
            # Croissance économique post-COVID
            elif year >= 2022:
                recovery = 1 + 0.025 * (year - 2022)
                df.loc[i, 'Import_Tonnes'] *= recovery
                df.loc[i, 'Export_Tonnes'] *= recovery
                df.loc[i, 'Passagers_Croisiere'] *= (1 + 0.15 * (year - 2022))
            
            # Développement du tourisme (2015+)
            if year >= 2015:
                tourism_growth = 1 + 0.03 * (year - 2015)
                df.loc[i, 'Passagers_Croisiere'] *= tourism_growth
            
            # Augmentation du commerce intra-caraïbes
            if year >= 2018:
                caribbean_growth = 1 + 0.04 * (year - 2018)
                df.loc[i, 'Caraibes_Import_Tonnes'] *= caribbean_growth
    
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
        
        # 6. Analyse saisonnière
        ax6 = plt.subplot(3, 2, 6)
        self._plot_seasonal_analysis(df, ax6)
        
        plt.suptitle(f'Analyse Import-Export - {self.port_name} (2010-2024)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('port_guadeloupe_trade_analysis.png', dpi=300, bbox_inches='tight')
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
                      'Produits_Petroliers_Tonnes', 'Vehicules_Tonnes',
                      'Sucre_Rhum_Tonnes', 'Bananes_Tonnes']
        labels = ['Alimentaire', 'Matériaux', 'Pétroliers', 'Véhicules', 'Sucre/Rhum', 'Bananes']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A602', '#6A0572', '#2A9D8F']
        
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
        partners = ['France_Import_Tonnes', 'Caraibes_Import_Tonnes', 
                   'Etats_Unis_Import_Tonnes', 'Europe_Import_Tonnes']
        labels = ['France', 'Caraïbes', 'États-Unis', 'Europe (hors France)']
        colors = ['#264653', '#2A9D8F', '#E76F51', '#F4A261']
        
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
            'Export_Tonnes': 'mean',
            'Passagers_Croisiere': 'mean'
        })
        
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        
        ax.plot(months, seasonal_data['Import_Tonnes']/1000, 
               label='Importations', marker='o', color='red')
        ax.plot(months, seasonal_data['Export_Tonnes']/1000, 
               label='Exportations', marker='o', color='green')
        
        ax2 = ax.twinx()
        ax2.plot(months, seasonal_data['Passagers_Croisiere']/1000, 
                label='Passagers Croisière (x1000)', marker='s', color='blue')
        
        ax.set_title('Analyse Saisonnière (moyenne mensuelle)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Tonnes (x1000)', color='black')
        ax2.set_ylabel('Passagers (x1000)', color='blue')
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
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
            'Alimentaire': df['Alimentaire_Tonnes'].mean(),
            'Matériaux': df['Materiaux_Tonnes'].mean(),
            'Pétroliers': df['Produits_Petroliers_Tonnes'].mean(),
            'Véhicules': df['Vehicules_Tonnes'].mean(),
            'Sucre/Rhum': df['Sucre_Rhum_Tonnes'].mean(),
            'Bananes': df['Bananes_Tonnes'].mean()
        }
        
        total = sum(commodities.values())
        for commodity, volume in commodities.items():
            percentage = (volume / total) * 100
            print(f"  {commodity}: {percentage:.1f}%")
        
        # 4. Partenaires commerciaux
        print("\n4. 🌍 PRINCIPAUX PARTENAIRES:")
        partners = {
            'France': df['France_Import_Tonnes'].mean(),
            'Caraïbes': df['Caraibes_Import_Tonnes'].mean(),
            'États-Unis': df['Etats_Unis_Import_Tonnes'].mean(),
            'Europe (hors France)': df['Europe_Import_Tonnes'].mean()
        }
        
        total_import = df['Import_Tonnes'].mean()
        for partner, volume in partners.items():
            percentage = (volume / total_import) * 100
            print(f"  {partner}: {percentage:.1f}% des importations")
        
        # 5. Tourisme de croisière
        print("\n5. 🚢 TOURISME DE CROISIÈRE:")
        avg_passengers = df['Passagers_Croisiere'].mean()
        max_passengers = df['Passagers_Croisiere'].max()
        print(f"Passagers moyens par mois: {avg_passengers:,.0f}")
        print(f"Record mensuel: {max_passengers:,.0f} passagers")
        
        # 6. Recommandations
        print("\n6. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        print("• Développer les exportations de produits agricoles (bananes, sucre, rhum)")
        print("• Renforcer les liaisons maritimes intra-caribéennes")
        print("• Capitaliser sur le tourisme de croisière comme moteur économique")
        print("• Diversifier les partenaires commerciaux au-delà de la France")
        print("• Investir dans des infrastructures portuaires modernes")
        print("• Promouvoir les produits guadeloupéens à l'exportation")

def main():
    """Fonction principale"""
    print("🏝️  ANALYSE IMPORT-EXPORT - PORT DE LA GUADELOUPE")
    print("=" * 60)
    
    # Initialiser l'analyseur
    analyzer = PortGuadeloupeTradeAnalyzer()
    
    # Générer les données
    trade_data = analyzer.generate_trade_data()
    
    # Sauvegarder les données
    output_file = 'port_guadeloupe_trade_data.csv'
    trade_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(trade_data[['Date', 'Import_Tonnes', 'Export_Tonnes', 'Balance_Commerciale', 'Passagers_Croisiere']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse commerciale...")
    analyzer.create_trade_analysis(trade_data)
    
    print(f"\n✅ Analyse du {analyzer.port_name} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Tonnage, valeur, partenaires, types de marchandises, tourisme de croisière")

if __name__ == "__main__":
    main()