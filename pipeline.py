import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

class BioprocessStatisticalControl:
    """
    Classe de surveillance statistique des procédés (SPC) utilisant l'ACP 
    pour détecter les anomalies et déviations de lots en bioréacteur.
    """
    def __init__(self, n_components=2, alpha=0.05):
        self.n_components = n_components
        self.alpha = alpha  # Seuil de signification statistique (95%)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.t2_threshold = None
        self.q_threshold = None

    def fit_normal_model(self, X_train):
        """
        Calcule le modèle de référence (Golden Batch) à partir de données normales.
        """
        # 1. Centrage et réduction des données (Standardisation)
        X_scaled = self.scaler.fit_transform(X_train)
        
        # 2. Application de l'ACP (Décomposition mathématique en valeurs propres)
        self.pca.fit(X_scaled)
        
        # 3. Calcul des métriques sur le modèle de référence
        scores = self.pca.transform(X_scaled)
        residuals = X_scaled - self.pca.inverse_transform(scores)
        
        # 4. Calcul du seuil théorique pour le T² de Hotelling (Distribution F de Fisher)
        n_samples, n_features = X_train.shape
        a = self.n_components
        f_stat = stats.f.ppf(1 - self.alpha, a, n_samples - a)
        self.t2_threshold = (a * (n_samples - 1) / (n_samples - a)) * f_stat
        
        # 5. Calcul du seuil théorique pour le Q-Residual (Approximation de Box)
        # Calcul des valeurs propres (eigenvalues) non conservées
        eigenvalues = self.pca.explained_variance_
        all_eigenvalues = np.var(X_scaled, axis=0) # Total variance par feature = 1 car standardisé
        # Variance résiduelle
        remaining_vars = np.sort(all_eigenvalues)[::-1][a:]
        
        theta1 = np.sum(remaining_vars)
        theta2 = np.sum(remaining_vars**2)
        theta3 = np.sum(remaining_vars**3)
        
        h0 = 1 - (2 * theta1 * theta3) / (3 * theta2**2)
        c_alpha = stats.norm.ppf(1 - self.alpha)
        
        # Formule mathématique de Jackson-Mudholkar pour le seuil Q
        self.q_threshold = theta1 * (
            (c_alpha * np.sqrt(2 * theta2 * h0**2) / theta1) 
            + 1 
            + (theta2 * h0 * (h0 - 1) / theta1**2)
        ) ** (1 / h0)

    def monitor_new_samples(self, X_new):
        """
        Évalue de nouvelles mesures de capteurs et renvoie le diagnostic statistique.
        """
        X_scaled = self.scaler.transform(X_new)
        scores = self.pca.transform(X_scaled)
        residuals = X_scaled - self.pca.inverse_transform(scores)
        
        # Calcul mathématique du T² de Hotelling pour chaque échantillon
        # T² = somme(score_i^2 / lambda_i)
        lambda_inv = np.diag(1.0 / self.pca.explained_variance_)
        t2_values = np.sum(scores.dot(lambda_inv) * scores, axis=1)
        
        # Calcul mathématique du Q-Residual (Somme des carrés des résidus)
        q_values = np.sum(residuals**2, axis=1)
        
        # Diagnostic
        t2_alarm = t2_values > self.t2_threshold
        q_alarm = q_values > self.q_threshold
        
        return pd.DataFrame({
            'Hotelling_T2': t2_values,
            'T2_Limit': self.t2_threshold,
            'T2_Alarm': t2_alarm,
            'Q_Residual': q_values,
            'Q_Limit': self.q_threshold,
            'Q_Alarm': q_alarm,
            'Process_Status': np.where(t2_alarm | q_alarm, 'DEVIATION', 'NORMAL')
        })

# --- SIMULATION POUR LE PROJET ---
if __name__ == "__main__":
    print("--- Simulation de données MSAT : Bioréacteur ---")
    np.random.seed(42)
    
    # 1. Simulation du "Golden Batch" (1000 points normaux : pH, Temp, O2, Glucose, Agitation)
    # Les variables biologiques sont fortement corrélées mathématiquement
    base_signal = np.random.normal(0, 1, (1000, 1))
    normal_data = np.hstack([
        base_signal + np.random.normal(7.0, 0.02, (1000, 1)),  # pH stable
        base_signal * 0.5 + np.random.normal(37.0, 0.1, (1000, 1)), # Temp (°C)
        -base_signal * 0.8 + np.random.normal(40.0, 1.5, (1000, 1)), # dO2 (%)
        -base_signal * 0.3 + np.random.normal(5.0, 0.2, (1000, 1))   # Glucose (g/L)
    ])
    
    # Entraînement du modèle statistique
    monitor = BioprocessStatisticalControl(n_components=2)
    monitor.fit_normal_model(normal_data)
    
    # 2. Simulation d'un lot avec anomalie (ex: rupture de la régulation de température et pH)
    anomalous_data = np.array([
        [7.01, 37.05, 40.2, 5.0], # Point normal
        [6.85, 38.90, 31.0, 4.2]  # Point critique : Dérive majeure détectée en QA/MSAT !
    ])
    
    results = monitor.monitor_new_samples(anomalous_data)
    print(results.to_string())
